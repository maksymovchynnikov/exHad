# Portable exhad build.  Prefer PYTHIA8_DIR, then a saved prefix,
# then pythia8-config on PATH.  The Pythia release is pinned to 8.317.

PYTHON ?= python3
CXX ?= c++
PYTHIA8_DIR ?=
PYTHIA8_REQUIRED_VERSION := 8.317
PYTHIA8_CONFIG ?= $(strip $(shell command -v pythia8-config 2>/dev/null))
BUILD_CONFIG := .pythia8-build

PYTHIA8_SAVED_PREFIX := $(strip $(shell test -f $(BUILD_CONFIG) && sed -n 's/^prefix=//p' $(BUILD_CONFIG)))
ifeq ($(strip $(PYTHIA8_DIR))$(strip $(PYTHIA8_CONFIG)),)
  PYTHIA8_DIR := $(PYTHIA8_SAVED_PREFIX)
endif

ifneq ($(strip $(PYTHIA8_DIR)),)
  PYTHIA8_PREFIX := $(abspath $(PYTHIA8_DIR))
  PYTHIA8_DIR_CONFIG := $(PYTHIA8_PREFIX)/bin/pythia8-config
  # A relocated package may ship a pythia8-config naming its build host;
  # use it only when it describes this prefix's headers, library and XML.
  PYTHIA8_DIR_CONFIG_OK := $(strip $(shell \
    if test -x "$(PYTHIA8_DIR_CONFIG)"; then \
      inc=`"$(PYTHIA8_DIR_CONFIG)" --includedir 2>/dev/null`; \
      lib=`"$(PYTHIA8_DIR_CONFIG)" --libdir 2>/dev/null`; \
      xml=`"$(PYTHIA8_DIR_CONFIG)" --xmldoc 2>/dev/null`; \
      test -f "$$inc/Pythia8/Pythia.h" && test -d "$$lib" && \
        test -f "$$xml/Version.xml" && echo yes; \
    fi))
  ifeq ($(PYTHIA8_DIR_CONFIG_OK),yes)
    PYTHIA8_CONFIG_USED := $(PYTHIA8_DIR_CONFIG)
  else
    PYTHIA8_CONFIG_USED :=
    PYTHIA8_LIBDIR := $(strip $(shell \
      for dir in "$(PYTHIA8_PREFIX)/lib" "$(PYTHIA8_PREFIX)/lib64"; do \
        ls "$$dir"/libpythia8.* >/dev/null 2>&1 && { echo "$$dir"; break; }; \
      done))
    PYTHIA8_LIBDIR := $(or $(PYTHIA8_LIBDIR),$(PYTHIA8_PREFIX)/lib)
    PYTHIA8_CXXFLAGS := -I$(PYTHIA8_PREFIX)/include
    PYTHIA8_LDFLAGS := -L$(PYTHIA8_LIBDIR) -Wl,-rpath,$(PYTHIA8_LIBDIR) -lpythia8 -ldl
    PYTHIA8_INCLUDEDIR := $(PYTHIA8_PREFIX)/include
    PYTHIA8_XMLDOC := $(PYTHIA8_PREFIX)/share/Pythia8/xmldoc
  endif
else ifneq ($(strip $(PYTHIA8_CONFIG)),)
  PYTHIA8_CONFIG_USED := $(abspath $(PYTHIA8_CONFIG))
  PYTHIA8_PREFIX := $(strip $(shell $(PYTHIA8_CONFIG_USED) --prefix))
else
  PYTHIA8_DISCOVERY_ERROR := no pythia8-config was found on PATH and PYTHIA8_DIR was not set
endif
ifneq ($(PYTHIA8_CONFIG_USED),)
  PYTHIA8_CXXFLAGS := $(shell $(PYTHIA8_CONFIG_USED) --cxxflags)
  PYTHIA8_LDFLAGS := $(shell $(PYTHIA8_CONFIG_USED) --ldflags)
  PYTHIA8_INCLUDEDIR := $(strip $(shell $(PYTHIA8_CONFIG_USED) --includedir))
  PYTHIA8_LIBDIR := $(strip $(shell $(PYTHIA8_CONFIG_USED) --libdir))
  PYTHIA8_XMLDOC := $(strip $(shell $(PYTHIA8_CONFIG_USED) --xmldoc))
endif

# Read the version from the XML payload itself, whichever discovery was used.
PYTHIA8_XML_VERSION := $(strip $(shell \
  test -f "$(PYTHIA8_XMLDOC)/Version.xml" && \
  sed -E -n 's/.*name="Pythia:versionNumber"[^>]*default="([^"]+)".*/\1/p' \
    "$(PYTHIA8_XMLDOC)/Version.xml" | head -n 1))

# Mandatory flags (threads: parallel Pythia initialization) stay separate from the conventional user variables.
EXHAD_CPPFLAGS := -Iinclude -pthread $(PYTHIA8_CXXFLAGS)
EXHAD_LDLIBS := -pthread $(PYTHIA8_LDFLAGS)
CPPFLAGS ?=
CXXFLAGS ?= -O2 -Wall -Wextra
CXXSTD := -std=c++17
LDFLAGS ?=
LDLIBS ?=

SRCS := src/main.cc src/isospin_cg.cc src/symmetry_filter.cc
OBJS := $(SRCS:.cc=.o)
DEPS := $(OBJS:.o=.d)
TARGET := exhad

.PHONY: all check preflight clean FORCE

all: $(TARGET) $(BUILD_CONFIG)

preflight:
	@if [ -n "$(PYTHIA8_DISCOVERY_ERROR)" ]; then \
	  echo "error: $(PYTHIA8_DISCOVERY_ERROR)" >&2; \
	  echo "hint: put pythia8-config on PATH, or run make PYTHIA8_DIR=/path/to/pythia8" >&2; \
	  exit 2; \
	fi
	@test -f "$(PYTHIA8_INCLUDEDIR)/Pythia8/Pythia.h" || { \
	  echo "error: Pythia header not found: $(PYTHIA8_INCLUDEDIR)/Pythia8/Pythia.h" >&2; exit 2; }
	@ls "$(PYTHIA8_LIBDIR)"/libpythia8.* >/dev/null 2>&1 || { \
	  echo "error: libpythia8 was not found in $(PYTHIA8_LIBDIR)" >&2; exit 2; }
	@for name in Version.xml Tunes.xml ParticleData.xml StringFragmentation.xml; do \
	  test -f "$(PYTHIA8_XMLDOC)/$$name" || { \
	    echo "error: Pythia runtime file not found: $(PYTHIA8_XMLDOC)/$$name" >&2; exit 2; }; \
	done
	@test "$(PYTHIA8_XML_VERSION)" = "$(PYTHIA8_REQUIRED_VERSION)" || { \
	  echo "error: exhad requires Pythia $(PYTHIA8_REQUIRED_VERSION), but $(PYTHIA8_XMLDOC)/Version.xml declares $(PYTHIA8_XML_VERSION)" >&2; \
	  exit 2; }

# The exact XML tree selected at build time, read by the Python launchers so
# a normal build never borrows EventCalc's different Pythia XML.
$(BUILD_CONFIG): FORCE | preflight
	@{ \
	  printf 'prefix=%s\n' "$(PYTHIA8_PREFIX)"; \
	  printf 'xmldoc=%s\n' "$(PYTHIA8_XMLDOC)"; \
	  printf 'libdir=%s\n' "$(PYTHIA8_LIBDIR)"; \
	  printf 'version=%s\n' "$(PYTHIA8_XML_VERSION)"; \
	  printf 'config=%s\n' "$(PYTHIA8_CONFIG_USED)"; \
	} > "$@.tmp"
	@cmp -s "$@.tmp" "$@" || mv "$@.tmp" "$@"
	@rm -f "$@.tmp"

$(OBJS): $(BUILD_CONFIG) $(wildcard include/*.h)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) $(EXHAD_LDLIBS) $(LDLIBS) -o $@

%.o: %.cc
	$(CXX) $(EXHAD_CPPFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXSTD) -MMD -MP -c $< -o $@

-include $(DEPS)

# End-to-end check: linkage, XML payload/version and one temporary event.
check: all
	@EXHAD_PYTHIA8DATA="$(PYTHIA8_XMLDOC)" $(PYTHON) ../tests/check_install.py

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET) $(BUILD_CONFIG) $(BUILD_CONFIG).tmp
