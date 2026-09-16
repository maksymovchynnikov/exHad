#include "exhad/Generator.hpp"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  try {
    exhad::Generator generator(argv[1], argv[2], argv[3], "central", "reference");
    double mass = std::stod(argv[4]);
    auto first = generator.generate(mass, 5, 123, exhad::DecayScope::All, {1., 0., 0.});
    auto second = generator.generate(mass, 5, 123, exhad::DecayScope::All, {1., 0., 0.});
    for (std::size_t i = 0; i < first.size(); ++i) {
      if (first[i].size() != second[i].size()) return 3;
      std::cout << "EVENT " << first[i].size() << '\n';
      for (std::size_t j = 0; j < first[i].size(); ++j) {
        const auto& p = first[i][j]; const auto& q = second[i][j];
        if (p.px != q.px || p.py != q.py || p.pz != q.pz || p.energy != q.energy || p.pdg != q.pdg)
          return 4;
        std::cout << std::setprecision(17) << p.px << ' ' << p.py << ' ' << p.pz
                  << ' ' << p.energy << ' ' << p.mass << ' ' << p.pdg << '\n';
      }
    }
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
