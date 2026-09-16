#!/usr/bin/env python3
"""Decay-channel fractions of every exHad model against the mass of the decaying particle.

    python examples/plot_channel_fractions.py --models dark-photon b-l --points 12 --events 2000
    python examples/plot_channel_fractions.py --nonhadronic --output-dir figures

One figure per model. Every point generates events with exHad itself, in parallel
worker processes, and sorts them into the channel groups of the supplementary
figures of the paper, so the curves carry the statistical error of the sample.
The default denominator is the hadronic width (exhad.Generator.generate). With
--nonhadronic the sample is the complete decay (generate_all), the denominator is
the total width, and the leptonic, radiative and invisible channels are drawn
under the labels the generator returns.

The generated events contain stable particles only, so the classifier rebuilds the
short-lived pi0, eta, K_S and eta' from the invariant masses of their daughters
before it counts hadrons. Rebuilt states are exact, but a parent whose daughters
are themselves misread (a Dalitz pi0, an eta in a busy event) moves to "Other";
this is a property of the figure, not of the generated events.

Models and mass ranges come from exhad.model_info, so renames or new models
need no edit here.
"""
import argparse
from collections import Counter
from itertools import chain, combinations, product
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'exhad-matplotlib'))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402

from exhad import Generator, model_info  # noqa: E402
from exhad.models import decays_with_hadrons  # noqa: E402
from exhad.core.pdg import m0  # noqa: E402
from exhad.models import MODELS  # noqa: E402

# Electron mixing of the HNL curves; the HNL channels depend on the mixing pattern.
HNL_MIXING = (1., 0., 0.)
# Rebuilt parents, in the order in which their daughters become available:
# pi0 and eta from photons, K_S from pions, eta' from an eta and pions.
CHAINS = ((111, ((22, 22),)),
          (221, ((22, 22), (211, -211, 111), (111, 111, 111), (211, -211, 22))),
          (310, ((211, -211), (111, 111))),
          (331, ((221, 211, -211), (221, 111, 111), (211, -211, 22))))
# Two-body daughters reconstruct their parent exactly; this window only absorbs
# the rounding of the event record and keeps accidental combinations away.
WINDOW_GEV = 2e-3
ORDER = ('pi', '2pi', '3pi', '4pi', '5pi', '6pi', 'k', 'kk', 'kkpi', 'kkpipi',
         'eta', 'etagamma', 'pi0gamma', 'nnbar', 'other')
LABELS = {'pi': r'$\pi$', 'k': r'$K$', '2pi': r'$2\pi$', '3pi': r'$3\pi$', '4pi': r'$4\pi$', '5pi': r'$5\pi$',
          '6pi': r'$6\pi$', 'kk': r'$K\bar K$', 'kkpi': r'$K\bar K\pi$',
          'kkpipi': r'$K\bar K\pi\pi$', 'eta': r'$\eta^{(\prime)}+X$',
          'etagamma': r'$\eta\gamma$', 'pi0gamma': r'$\pi^0\gamma$',
          'nnbar': r'$N\bar N$', 'other': 'Other'}
# Hues of the supplementary figures; 6pi and the nonhadronic channels extend them.
COLORS = {'pi': '#00A6A6', 'k': '#6A3D9A', '2pi': '#2255B5', '3pi': '#D55E00', '4pi': '#009E73', '5pi': '#7B2CBF',
          '6pi': '#B22222', 'kk': '#A54C92', 'kkpi': '#E69F00', 'kkpipi': '#56B4E9',
          'eta': '#617A24', 'etagamma': '#8C6D1F', 'pi0gamma': '#CC79A7',
          'nnbar': '#7C4A2D', 'other': '#222222'}
EXTRA_COLORS = ('#2255B5', '#D55E00', '#009E73', '#7B2CBF', '#E69F00', '#56B4E9',
                '#CC79A7', '#617A24', '#7C4A2D', '#B22222', '#A54C92', '#8C6D1F')
RATE_COLOR, START_COLOR = '#087F8C', '#AD3A93'
DASH = (0, (4, 2.5))


def invariant_mass(particles):
    px, py, pz, energy = (sum(p[i] for p in particles) for i in range(1, 5))
    return max(energy * energy - px * px - py * py - pz * pz, 0.) ** .5


def combinations_of(particles, pattern):
    """Index tuples of the particles that carry exactly the PDG codes of pattern."""
    pools, by_pdg = [], {}
    for index, particle in enumerate(particles):
        by_pdg.setdefault(particle[0], []).append(index)
    for pdg, needed in Counter(pattern).items():
        if len(by_pdg.get(pdg, ())) < needed:
            return
        pools.append(combinations(by_pdg[pdg], needed))
    for choice in product(*pools):
        yield tuple(chain.from_iterable(choice))


def absorb(particles, pdg, patterns):
    """Replace daughter sets whose invariant mass is the parent's by that parent.

    Every candidate is scored once and the closest disjoint ones are taken first,
    so a photon pair that is a pi0 is not spent on a worse eta candidate.
    """
    mass, found = m0(pdg), []
    for pattern in patterns:
        for indices in combinations_of(particles, pattern):
            residual = abs(invariant_mass([particles[i] for i in indices]) - mass)
            if residual < WINDOW_GEV:
                found.append((residual, indices))
    used, parents = set(), []
    for _, indices in sorted(found, key=lambda item: item[0]):
        if used.isdisjoint(indices):
            used.update(indices)
            daughters = [particles[i] for i in indices]
            parents.append((pdg, *(sum(d[i] for d in daughters) for i in range(1, 5))))
    return [p for i, p in enumerate(particles) if i not in used] + parents


def hadrons(event):
    """PDG counts of one event after the short-lived parents are rebuilt.

    Leptons and neutrinos are left out: they are spectators of the hadronic
    channel, such as the charged lepton or the neutrino of an HNL decay.
    """
    particles = [(int(p[5]), p[0], p[1], p[2], p[3]) for p in event
                 if abs(int(p[5])) not in (11, 12, 13, 14, 15, 16)]
    for pdg, patterns in CHAINS:
        particles = absorb(particles, pdg, patterns)
    return Counter(abs(p[0]) if p[0] in (-211, -321, -2212, -2112) else p[0] for p in particles)


def channel_group(counts):
    """The channel group of one hadronic event: the groups of the paper's figures."""
    pions = counts[211] + counts[111]
    kaons = counts[321] + counts[310] + counts[130]
    nucleons = counts[2212] + counts[2112]
    etas = counts[221] + counts[331]
    photons, total = counts[22], sum(counts.values())
    if total != pions + kaons + nucleons + etas + photons:
        return 'other'  # leptons, neutrinos or charm and bottom hadrons
    if nucleons >= 2:  # a nucleon pair with further hadrons is not the exclusive pair
        return 'nnbar' if total == 2 else 'other'
    if etas:
        return 'etagamma' if total == 2 and etas == 1 and photons == 1 else 'eta'
    if kaons >= 2 and not photons:
        return {0: 'kk', 1: 'kkpi', 2: 'kkpipi'}.get(pions, 'other')
    if total == 1:  # a single pion or kaon, as in N -> pi e and N -> K e
        return 'pi' if pions else 'k' if kaons else 'other'
    if total == 2 and photons == 1 and counts[111] == 1:
        return 'pi0gamma'
    if not kaons and not photons and 2 <= pions <= 6:
        return f'{pions}pi'
    return 'other'


def masses_of(info, nonhadronic, points):
    """Mass points over the generation range of the model, spaced logarithmically."""
    low, high = info['generation_gev']['all' if nonhadronic else decays_with_hadrons(info['model'])]
    return [round(float(mass), 6) for mass in np.geomspace(low * 1.02, high * .999, points)]


def fractions_at(generator, model, mass, events, seed, nonhadronic):
    mixing = list(HNL_MIXING) if model == 'hnl' else None
    if nonhadronic:
        reply = generator.generate_all(mass, events, seed=seed, mixing=mixing)
        sample = list(zip(reply['channel_labels'], reply['events']))
    else:
        sample = [('hadronic', event) for event in generator.generate(mass, events, seed=seed, mixing=mixing)]
    # An event with hadrons belongs to a channel group whatever the generator calls
    # its row; a decay without hadrons keeps the label, which names its leptons.
    counts = Counter()
    for label, event in sample:
        content = hadrons(event)
        counts[channel_group(content) if content else label] += 1
    return {group: count / len(sample) for group, count in counts.items()}, len(sample)


def scan(model, points, events, workers, seed, nonhadronic):
    """Channel fractions of one model at every mass point; one worker pool for all points."""
    info = model_info(model)
    rows = []
    with Generator(model, workers=workers) as generator:
        for index, mass in enumerate(masses_of(info, nonhadronic, points)):
            try:
                fractions, generated = fractions_at(generator, model, mass, events, seed + index, nonhadronic)
            except RuntimeError as error:
                # The lowest hadronic rows open above the lightest hadron; such a
                # point carries no decay of this kind and is left out of the curve.
                if 'no open hadronic channel' not in str(error):
                    raise
                print(f'{model}: no open hadronic channel at {mass} GeV', file=sys.stderr)
                continue
            rows.append({'mass_gev': mass, 'events': generated, 'fractions': fractions})
    return {'model': model, 'mother_pdg': info['mother_pdg'], 'nonhadronic': nonhadronic,
            'mixing': list(HNL_MIXING) if model == 'hnl' else None,
            'support_gev': info['support_gev'], 'exclusive_gev': info.get('exclusive_gev'),
            'points': rows}


def curves(scan_result, most=6):
    """Drawn groups in the order of the figures, hadronic groups first.

    Only the most prominent nonhadronic labels get a curve of their own; the
    rest are summed into one curve, which keeps the legend readable for models
    with many leptonic rows.
    """
    rows = scan_result['points']
    largest = {group: max(row['fractions'].get(group, 0.) for row in rows)
               for group in {group for row in rows for group in row['fractions']}}
    seen = {group for group, value in largest.items() if value > 0}
    ordered = [group for group in ORDER if group in seen]
    extra = sorted(seen - set(ORDER), key=lambda group: -largest[group])
    merged, extra = extra[most:], extra[:most]
    colors = dict(COLORS, **{group: EXTRA_COLORS[i % len(EXTRA_COLORS)] for i, group in enumerate(extra)})
    masses = [row['mass_gev'] for row in rows]
    for group in ordered:
        values = [row['fractions'].get(group, 0.) or np.nan for row in rows]
        yield LABELS[group], colors[group], '-', masses, values
    for group in extra:  # dashed: a row of the decay table, not a channel group
        values = [row['fractions'].get(group, 0.) or np.nan for row in rows]
        yield group, colors[group], DASH, masses, values
    if merged:
        values = [sum(row['fractions'].get(group, 0.) for group in merged) or np.nan for row in rows]
        yield f'{len(merged)} further rows', '#777777', DASH, masses, values


def draw(scan_result, path):
    model, nonhadronic = scan_result['model'], scan_result['nonhadronic']
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 18,
                         'axes.labelsize': 22, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    fig.subplots_adjust(left=.115, right=.97, bottom=.16, top=.975)
    handles = []
    for label, color, style, masses, values in curves(scan_result):
        handles.append(Line2D([], [], color=color, lw=2.3, ls=style, label=label))
        ax.plot(masses, values, color=color, lw=2.4, ls=style)
    masses = [row['mass_gev'] for row in scan_result['points']]
    denominator = r'\Gamma_{\mathrm{tot}}' if nonhadronic else r'\Gamma_{\mathrm{had,tot}}'
    # The top strip above one carries the legends; fractions cannot reach it.
    ax.set(xscale='log', yscale='log', xlim=(masses[0] * .95, masses[-1] * 1.05), ylim=(1e-4, 100.),
           xlabel=r'$m\ [\mathrm{GeV}]$', ylabel=rf'$\Gamma_{{\mathrm{{channel}}}}/{denominator}$')
    decades = (.0001, .001, .01, .1, 1, 10, 100)
    fine = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2, 5, 10, 20, 50)
    # Every second decade would still collide over a range this wide.
    scale = decades if masses[-1] > 100 * masses[0] else fine
    ticks = [value for value in scale if masses[0] * .95 <= value <= masses[-1] * 1.05]
    ax.set_xticks(ticks, minor=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, _: ''))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    title = model + (r', $|U_e|^2$ only' if model == 'hnl' else '')
    ax.text(.035, .97, title, ha='left', va='top', transform=ax.transAxes, fontsize=17,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.2})
    channels = ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(.99, .83), ncol=4,
                         columnspacing=.55, handlelength=1., handletextpad=.35, labelspacing=.20,
                         borderpad=.25, framealpha=.65, facecolor='white', edgecolor='none',
                         fontsize=13.5)
    ax.add_artist(channels)  # a second legend would otherwise replace the channels
    markers = transitions(ax, scan_result)
    if markers:
        ax.legend(handles=markers, loc='upper right', bbox_to_anchor=(.99, .94), ncol=2,
                  columnspacing=.6, handlelength=1.5, fontsize=12.5, framealpha=.8,
                  facecolor='white', edgecolor='none')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(path.with_suffix('.png'), dpi=150)
    plt.close(fig)


def transitions(ax, scan_result):
    """Shade where the hadronic decays are tabulated channels and mark where fragmentation starts."""
    exclusive, support = scan_result['exclusive_gev'], scan_result['support_gev']
    handles = []
    if exclusive:
        ax.axvspan(*exclusive, color=RATE_COLOR, alpha=.055, linewidth=0, zorder=.1)
        ax.axvline(exclusive[1], color=START_COLOR, linestyle=(0, (4, 2.5)), linewidth=1.5, zorder=.4)
        handles = [Patch(facecolor=RATE_COLOR, edgecolor=RATE_COLOR, alpha=.40, label='Exclusive channels'),
                   Line2D([], [], color=START_COLOR, linestyle=(0, (4, 2.5)), linewidth=1.5,
                          label='Fragmentation starts')]
    elif support:  # the HNL has no exclusive table: its matched interval is the W routes
        ax.axvspan(*support, color=RATE_COLOR, alpha=.055, linewidth=0, zorder=.1)
        handles = [Patch(facecolor=RATE_COLOR, edgecolor=RATE_COLOR, alpha=.40, label=r'Matched $W$ routes')]
    return handles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS),
                        metavar='MODEL', help='models to plot (default: all)')
    parser.add_argument('--points', type=int, default=24, help='mass points per model')
    parser.add_argument('--events', type=int, default=2000, help='events per mass point')
    parser.add_argument('--workers', type=int, help='worker processes (default: one per core, at most 8)')
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    parser.add_argument('--nonhadronic', action='store_true',
                        help='include the nonhadronic decays and normalize to the total width')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(argv)
    if args.points < 2 or args.events < 1:
        parser.error('--points needs at least 2 and --events at least 1')
    suffix = '_all' if args.nonhadronic else ''
    summary = {}
    for model in args.models:
        started = time.monotonic()
        result = scan(model, args.points, args.events, args.workers, args.seed, args.nonhadronic)
        path = args.output_dir / f'channel_fractions_{model}{suffix}.pdf'
        draw(result, path)
        result['seconds'] = round(time.monotonic() - started, 1)
        (args.output_dir / f'channel_fractions_{model}{suffix}.json').write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        summary[model] = {'figure': str(path), 'seconds': result['seconds'],
                          'points': len(result['points'])}
        print(json.dumps({model: summary[model]}), flush=True)
    print(json.dumps({'figures': summary}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
