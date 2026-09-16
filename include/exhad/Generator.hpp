#pragma once
// C++17 client for the complete exHad model. Linux/macOS, no Python linkage.
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace exhad {
struct Particle {
  double px, py, pz, energy, mass;
  int pdg;
};

using Event = std::vector<Particle>;
enum class DecayScope { Hadronic, All };

// PDG identifier of the model's decaying particle, as exhad.models.MOTHER_PDG.
inline int defaultMotherPdg(const std::string& model) {
  if (model == "dark-photon") return 4900022;
  if (model == "alp-fermion") return 36;
  if (model == "hnl") return 9900012;
  if (model.rfind("scalar", 0) == 0) return 35;
  if (model.rfind("b-l", 0) == 0) return 32;
  throw std::invalid_argument("No default mother PDG identifier for model " + model);
}

// One instance per calling thread. The persistent subprocess isolates the
// model's pinned Pythia from the experiment's Pythia library and Python state.
// It generates each request in chunks of chunkSize events on `workers` worker
// processes (0: one per physical core, at most 8; 1: a single process). Events
// depend on the seed, count and chunkSize, never on the number of workers.
class Generator {
 public:
  Generator(const std::string& root, const std::string& python,
            std::string model, std::string variation = "central",
            std::string execution = "auto", int workers = 0, std::size_t chunkSize = 512,
            int timeoutSeconds = 600)
      : timeoutMs_(0), workers_(workers), chunkSize_(chunkSize), model_(std::move(model)),
        variation_(std::move(variation)), execution_(std::move(execution)) {
    token(model_);
    token(variation_);
    token(execution_);
    if (root.empty() || root.front() != '/' || python.empty() || python.front() != '/')
      throw std::invalid_argument("Use absolute exHad and Python paths");
    if (workers < 0 || workers > 64) throw std::invalid_argument("workers must be 0..64");
    if (chunkSize < 1 || chunkSize > 1000000) throw std::invalid_argument("chunkSize must be 1..1000000");
    if (timeoutSeconds < 1 || timeoutSeconds > 86400)
      throw std::invalid_argument("timeoutSeconds must be 1..86400");
    timeoutMs_ = timeoutSeconds * 1000;
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair)) fail("socketpair");
    for (int i = 0; i < 2; ++i) {
      if (pair[i] <= STDERR_FILENO) {
        int fd = fcntl(pair[i], F_DUPFD, 3);
        if (fd < 0) {
          ::close(pair[0]);
          ::close(pair[1]);
          fail("fcntl");
        }
        ::close(pair[i]);
        pair[i] = fd;
      }
      fcntl(pair[i], F_SETFD, FD_CLOEXEC);
    }
#ifdef SO_NOSIGPIPE
    int one = 1;
    setsockopt(pair[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pair[1], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, pair[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pair[0]);
    posix_spawn_file_actions_addclose(&actions, pair[1]);
    std::string script = root + "/exhad/worker.py";
    char* argv[] = {const_cast<char*>(python.c_str()), const_cast<char*>(script.c_str()),
                    const_cast<char*>("--cpp"), nullptr};
    int code = posix_spawn(&pid_, python.c_str(), &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(pair[1]);
    if (code) {
      ::close(pair[0]);
      pid_ = -1;
      fail("cannot launch Python worker");
    }
    fd_ = pair[0];
  }
  Generator(const Generator&) = delete;
  Generator& operator= (const Generator&) = delete;
  ~Generator() {
    close();
  }
  const std::string& model() const {
    return model_;
  }

  std::vector<Event> generate(double mass, std::size_t count, std::uint64_t seed = 1,
      DecayScope scope = DecayScope::Hadronic,
      std::array<double, 3> mixingSquared = {0., 0., 0.}) {
    return exchange(header(mass, count, seed, scope == DecayScope::All ? "all" : "hadronic", mixingSquared) + '\n', count);
  }

  // Complete decays of named explicit rows of the model's EventCalc-format decay table, as
  // Generator.generate_rows in Python: one event list per (label, count), in that order.
  // terminal is "pythia" or "matched" (taus decay fully in both); HNL requires the squared mixings.
  std::vector<std::vector<Event>> generateRows(double mass,
      const std::vector<std::pair<std::string, std::size_t>>& rows, std::uint64_t seed = 1,
      const std::string& terminal = "pythia", std::array<double, 3> mixingSquared = {0., 0., 0.}) {
    std::size_t count = 0;
    std::string suffix = ' ' + terminal;
    token(terminal);
    for (const auto& row : rows) {
      token(row.first);
      count += row.second;
      suffix += ' ' + row.first + '=' + std::to_string(row.second);
    }
    auto events = exchange(header(mass, count, seed, "rows", mixingSquared) + suffix + '\n', count);
    std::vector<std::vector<Event>> result;
    auto next = events.begin();
    for (const auto& row : rows) {
      result.emplace_back(std::make_move_iterator(next), std::make_move_iterator(next + row.second));
      next += row.second;
    }
    return result;
  }

  void close() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (pid_ <= 0) return;
    // EOF lets the worker close its native generators normally.
    for (int i = 0; i < 1000; ++i) {
      int result = waitpid(pid_, nullptr, WNOHANG);
      if (result == pid_ || (result < 0 && errno == ECHILD)) {
        pid_ = -1;
        return;
      }
      poll(nullptr, 0, 10);
    }
    kill(pid_, SIGKILL);
    while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
    pid_ = -1;
  }

 private:
  std::string header(double mass, std::size_t count, std::uint64_t seed, const char* scope,
                     const std::array<double, 3>& mixingSquared) const {
    if (!std::isfinite(mass) || mass <= 0 || count > 1000000)
      throw std::invalid_argument("Invalid mass or count (maximum batch: 1000000)");
    std::ostringstream request;
    request.imbue(std::locale::classic());
    request << std::setprecision(17) << model_ << ' ' << variation_ << ' ' << execution_ << ' '
            << workers_ << ' ' << chunkSize_ << ' ' << mass << ' ' << count << ' ' << seed
            << ' ' << scope;
    for (double x : mixingSquared) {
      if (!std::isfinite(x) || x < 0) throw std::invalid_argument("Invalid squared mixing");
      request << ' ' << x;
    }
    return request.str();
  }

  std::vector<Event> exchange(const std::string& request, std::size_t count) {
    try {
      send(request);
      // Exact token counts: trailing text (such as the ".5" of a PDG "211.5") is malformed.
      auto header = fields(readLine());
      std::string tag, extra;
      std::size_t received;
      if (!(header >> tag >> received) || (header >> extra) || tag != "OK" || received != count)
        fail("wrong batch header");
      std::vector<Event> events(count);
      for (auto& event : events) {
        auto row = fields(readLine());
        std::size_t n;
        if (!(row >> tag >> n) || (row >> extra) || tag != "EVENT" || n < 1 || n > 100000)
          fail("wrong event header");
        event.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
          Particle p;
          auto values = fields(readLine());
          if (!(values >> p.px >> p.py >> p.pz >> p.energy >> p.mass >> p.pdg) || (values >> extra)
              || !std::isfinite(p.px) || !std::isfinite(p.py) || !std::isfinite(p.pz)
              || !std::isfinite(p.energy) || !std::isfinite(p.mass)
              || p.energy < 0 || p.mass < 0 || p.pdg == 0) fail("invalid particle record");
          event.push_back(p);
        }
      }
      if (readLine() != "END") fail("missing batch terminator");
      return events;
    } catch (...) {
      close();
      throw;
    }
  }

  int fd_ = -1, timeoutMs_, workers_;
  std::size_t chunkSize_;
  pid_t pid_ = -1;
  std::string model_, variation_, execution_, buffer_;
  [[noreturn]] static void fail(const std::string& what) {
    throw std::runtime_error("exHad: " + what);
  }
  static void token(const std::string& s) {
    if (s.empty() || s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.") != std::string::npos)
      throw std::invalid_argument("Invalid model/variation/execution name");
  }
  static std::istringstream fields(const std::string& s) {
    if (s.rfind("ERROR ", 0) == 0) fail(s.substr(6));
    std::istringstream result(s);
    result.imbue(std::locale::classic());
    return result;
  }
  void send(const std::string& message) {
    if (fd_ < 0) fail("generator is closed");
    std::size_t offset = 0;
    while (offset < message.size()) {
      int flags = 0;
#ifdef MSG_NOSIGNAL
      flags = MSG_NOSIGNAL;
#endif
      auto n = ::send(fd_, message.data() + offset, message.size() - offset, flags);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) fail("worker connection closed");
      offset += n;
    }
  }
  std::string readLine() {
    for (;;) {
      auto end = buffer_.find('\n');
      if (end != std::string::npos) {
        std::string line = buffer_.substr(0, end);
        buffer_.erase(0, end + 1);
        return line;
      }
      pollfd wait{fd_, POLLIN, 0};
      int ready = poll(&wait, 1, timeoutMs_);
      if (ready < 0 && errno == EINTR) continue;
      if (ready <= 0) fail("worker failed or timed out; no Pythia fallback was used");
      char bytes[8192];
      auto n = recv(fd_, bytes, sizeof(bytes), 0);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) fail("worker exited; see its stderr diagnostics");
      buffer_.append(bytes, n);
      if (buffer_.size() > 1000000) fail("invalid worker response");
    }
  }
};
}  // namespace exhad
