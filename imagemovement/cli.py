"""Command-line interface for image-movement.

  compare A B   run the cascade on one image pair and print the evidence
  eval          run the three-category harness and print precision/recall

Tunable config is read from DetectorConfig (env-overridable, e.g.
IMOVE_STAGE2__MIN_INLIERS=30), so this stays a thin wrapper over the library.
"""

from __future__ import annotations

import argparse
import sys

from .config import DetectorConfig
from .data import load_image, synthetic_seeds
from .evaluate import evaluate, format_report
from .stage1_hash import hamming, phash
from .stage2_geom import GeoVerifier


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = DetectorConfig()
    a = load_image(args.image_a)
    b = load_image(args.image_b)
    dist = hamming(phash(a, cfg.stage1.hash_size), phash(b, cfg.stage1.hash_size))
    verifier = GeoVerifier(cfg.stage2)
    ev = verifier.verify(a, b)
    match = verifier.is_match(ev)
    print(f"hash distance:    {dist}  (filter passes <= {cfg.stage1.max_distance})")
    print(f"keypoint matches: {ev.matches}")
    print(f"RANSAC inliers:   {ev.inliers}  (need >= {cfg.stage2.min_inliers})")
    print(f"est. zoom:        {ev.scale:.3f}")
    print(f"est. rotation:    {ev.rotation_deg:.2f} deg")
    print(f"est. shift:       {ev.translation:.1f} px")
    print(f"residual:         {ev.residual:.1f}  (photometric gate passes <= {cfg.stage2.max_residual})")
    print(f"all gates pass:   {ev.geom_ok}")
    print(f"VERDICT:          {'SAME IMAGE (match)' if match else 'different images (no match)'}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = DetectorConfig()
    if args.dataset == "synthetic":
        identities = [(name, [img]) for name, img in synthetic_seeds(cfg.eval.synthetic_seeds)]
        source = f"{len(identities)} synthetic single-image identities (no hard negatives)"
    elif args.dataset == "celeba":
        from .datasets import load_celeba
        identities = load_celeba(n_images=args.identities)
        source = (f"{len(identities)} CelebA faces @178x218 "
                  f"(each its own identity; same-person test is LFW-only)")
    else:  # lfw
        from .datasets import load_lfw
        identities = load_lfw(max_identities=args.identities)
        source = f"{len(identities)} LFW identities (real faces, multi-photo)"
    print(f"data source: {source}\n")
    print(format_report(evaluate(identities, cfg)))
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    from .service import ReuseDetectorService
    svc = ReuseDetectorService(DetectorConfig())
    rec = svc.enroll(load_image(args.image), args.user, args.attempt)
    print(f"enrolled record {rec.id}  (user={rec.user_id}, attempt={rec.attempt_id})  corpus size={len(svc.corpus)}")
    svc.close()
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .service import ReuseDetectorService
    svc = ReuseDetectorService(DetectorConfig())
    matches = svc.check(load_image(args.image))
    alert = svc.alert(matches)
    if not matches:
        print("no reuse detected (0 matches)")
    else:
        print(f"REUSE DETECTED: {len(matches)} match(es) across {alert.distinct_users} distinct user(s)  "
              f"-> alert={'TRIGGERED' if alert.triggered else 'below threshold'} (severity={alert.severity:.2f})")
        for m in matches:
            print(f"  record {m.record_id}  user={m.user_id}  attempt={m.attempt_id}  "
                  f"inliers={m.evidence.inliers}  residual={m.evidence.residual:.1f}  hash_dist={m.hash_distance}")
    svc.close()
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    from .service import ReuseDetectorService
    svc = ReuseDetectorService(DetectorConfig())
    alert, rec = svc.submit(load_image(args.image), args.user, args.attempt)
    print(f"enrolled record {rec.id}  (user={rec.user_id}, attempt={rec.attempt_id})  corpus size={len(svc.corpus)}")
    if not alert.matches:
        print("no prior reuse -- clean.")
    else:
        print(f"REUSE AT INTAKE: {len(alert.matches)} match(es) across {alert.distinct_users} distinct user(s)  "
              f"-> alert={'TRIGGERED' if alert.triggered else 'below threshold'} (severity={alert.severity:.2f})")
        for m in alert.matches:
            print(f"  record {m.record_id}  user={m.user_id}  attempt={m.attempt_id}  "
                  f"inliers={m.evidence.inliers}  residual={m.evidence.residual:.1f}")
    svc.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve
    print(f"serving image-movement API on http://{args.host}:{args.port}  (interactive console at /docs)")
    serve(host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imagemovement", description="image-movement detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cmp = sub.add_parser("compare", help="compare two images and print the evidence")
    p_cmp.add_argument("image_a")
    p_cmp.add_argument("image_b")
    p_cmp.set_defaults(func=cmd_compare)

    p_eval = sub.add_parser("eval", help="run the precision/recall harness")
    p_eval.add_argument("--dataset", choices=["lfw", "celeba", "synthetic"], default="lfw",
                        help="validation dataset (default: lfw)")
    p_eval.add_argument("--identities", type=int, default=25,
                        help="LFW identities, or CelebA image count")
    p_eval.set_defaults(func=cmd_eval)

    p_enroll = sub.add_parser("enroll", help="enroll an image into the corpus")
    p_enroll.add_argument("image")
    p_enroll.add_argument("--user", required=True, help="user_id submitting the image")
    p_enroll.add_argument("--attempt", default="-", help="attempt_id for this image")
    p_enroll.set_defaults(func=cmd_enroll)

    p_check = sub.add_parser("check", help="check an image for reuse against the corpus")
    p_check.add_argument("image")
    p_check.set_defaults(func=cmd_check)

    p_submit = sub.add_parser("submit", help="check an image for reuse AND enroll it (intake flow)")
    p_submit.add_argument("image")
    p_submit.add_argument("--user", required=True, help="user_id submitting the image")
    p_submit.add_argument("--attempt", default="-", help="attempt_id for this image")
    p_submit.set_defaults(func=cmd_submit)

    p_serve = sub.add_parser("serve", help="run the HTTP API (live demo + /docs console)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
