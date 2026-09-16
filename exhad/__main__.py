import argparse
import math
from pathlib import Path
from . import Generator
from .output import check_format, write_events
from .models import MODELS, decays_with_hadrons, mother_pdg


def main():
    parser = argparse.ArgumentParser(description='Generate exHad rest-frame decays')
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--mass', type=float, required=True)
    parser.add_argument('--events', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--variation', default='central')
    parser.add_argument('--all-decays', action='store_true',
                        help='Generate the full model: every decay channel, not only the decays with hadrons')
    parser.add_argument('--mixing', nargs=3, type=float, metavar=('UE2', 'UMU2', 'UTAU2'),
                        help='Relative squared HNL mixings; required for HNL')
    parser.add_argument('--weighted', action='store_true', help='alp-fermion importance sampling with explicit event weights')
    parser.add_argument('--weight-floor-fraction', type=float, default=.1)
    parser.add_argument('--workers', type=int, default=None,
                        help='Worker processes (default: one per physical core, at most 8); events do not depend on it')
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=('json', 'hepmc', 'hepmc2'), default='json',
                        help='Output format: JSON, HepMC3 ASCII, or HepMC2 ASCII')
    parser.add_argument('--mother-pdg', type=int,
                        help='PDG identifier of the mother (default: the model\'s, see exhad.models.MOTHER_PDG)')
    args = parser.parse_args()

    if (args.model == 'hnl') != (args.mixing is not None):
        parser.error('HNL generation requires --mixing UE2 UMU2 UTAU2, which applies only to HNL')

    if args.output.exists():
        parser.error('output exists; choose a new file')

    try:
        mother = mother_pdg(args.model, args.mother_pdg)
        check_format(args.format)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    with Generator(args.model, workers=args.workers, chunk_size=args.chunk_size,
                   variation=args.variation) as generator:
        if args.all_decays:
            reply = generator.generate_all(args.mass, args.events, seed=args.seed,
                mixing=args.mixing, weighted=args.weighted,
                weight_floor_fraction=args.weight_floor_fraction)
        elif args.weighted:
            reply = generator.generate_weighted(args.mass, args.events, seed=args.seed,
                weight_floor_fraction=args.weight_floor_fraction)
        else:
            reply = dict(events=generator.generate(args.mass, args.events, seed=args.seed,
                                                   **({'mixing': args.mixing} if args.mixing is not None else {})))
    events = reply['events']
    payload = {'schema': 'exhad-rest-frame-events-v1', 'model': args.model, 'mother_pdg': mother,
               'mass_gev': args.mass, 'seed': args.seed, 'variation': args.variation,
               'particle_fields': ['px', 'py', 'pz', 'E', 'mass', 'PDG'], 'events': events}
    payload['execution'] = dict(chunk_size=args.chunk_size)
    payload['decay_scope'] = 'all' if args.all_decays else decays_with_hadrons(args.model)
    payload.update(reply)

    if args.mixing is not None:
        payload['mixing_squared'] = args.mixing

    if args.weighted:
        active = [group == 'fragmentation' for group in reply['normalization_groups']]
        total = math.fsum(w for w, a in zip(reply['raw_weights'], active) if a)
        factor = sum(active) / total if any(active) else 1.
        payload.update(schema='exhad-weighted-rest-frame-events-v1', fragmentation_normalization=factor,
            hadronization_weights=[w * factor if a else w for w, a in zip(reply['raw_weights'], active)],
            normalization_scope='This entire mass/variation sample, before analysis selections')
    write_events(args.output, payload, format=args.format, mother_pdg=mother)
    print(f'Wrote {len(events)} {payload["decay_scope"]}-decay events to {args.output}')


if __name__ == '__main__':
    main()
