"""Separate stems that land on one point (fusion.resolve_stacks)."""

import numpy as np

from memoryscapes_workers.audio.fusion import Candidate, Cluster, Placement, fuse, resolve_stacks

VEHICLE = np.array([1.9, 0.5, 5.1])
KIOSK = np.array([-3.0, 0.8, 7.0])


def cluster(centre, weight=1.0, count=4):
    return Cluster(centre=np.asarray(centre, float), weight=weight, spread_m=0.3, count=count)


def candidate(sid, clusters):
    return Candidate(sid, fuse(clusters, None, None), clusters, None, None, agree_radius_m=2.0)


def test_a_stem_with_somewhere_else_to_go_moves_there():
    """The siren's prompt also found the vehicle the horn is on, but most of its
    detections are on the kiosk speaker: it goes there, the horn keeps the vehicle."""
    horn = candidate("s1_car_horn", [cluster(VEHICLE, 3.0)])
    siren = candidate("s2_siren", [cluster(VEHICLE, 1.0), cluster(KIOSK, 0.9)])
    assert np.linalg.norm(siren.placement.position - VEHICLE) < 0.1  # stacked before
    resolve_stacks([horn, siren])
    assert np.allclose(horn.placement.position, VEHICLE) and horn.shares_with is None
    assert np.allclose(siren.placement.position, KIOSK) and siren.shares_with is None


def test_a_stem_with_nowhere_else_shares_the_object_exactly():
    """TestVideo: siren and car horn both found only the one vehicle. They are taken
    as one object making both sounds: one exact point, the weaker naming the keeper."""
    horn = candidate("s5_car_horn", [cluster(VEHICLE + [0.0, 0.0, 0.1], 7.0, count=7)])
    siren = candidate("s4_siren", [cluster(VEHICLE, 7.0, count=7), cluster([2.9, 1.6, 5.7], 1.0, count=1)])
    horn.placement.confidence, siren.placement.confidence = 0.65, 0.61
    resolve_stacks([siren, horn])
    assert horn.shares_with is None and siren.shares_with == "s5_car_horn"
    assert np.array_equal(siren.placement.position, horn.placement.position)
    assert siren.placement.confidence == 0.61  # its own confidence, not the keeper's


def test_sources_apart_and_diffuse_ones_are_left_alone():
    a = candidate("a", [cluster(VEHICLE)])
    b = candidate("b", [cluster(KIOSK)])
    diffuse = Candidate("c", Placement(None, None, 0.0, "diffuse"), [], None, None, 2.0)
    resolve_stacks([a, b, diffuse])
    assert np.allclose(a.placement.position, VEHICLE) and np.allclose(b.placement.position, KIOSK)
    assert diffuse.placement.position is None and not any(x.shares_with for x in (a, b, diffuse))


def test_three_on_one_spot_all_join_the_most_confident():
    stems = [candidate(f"s{i}", [cluster(VEHICLE + [0.1 * i, 0, 0])]) for i in range(3)]
    for i, s in enumerate(stems):
        s.placement.confidence = [0.4, 0.7, 0.5][i]
    resolve_stacks(stems)
    assert [s.shares_with for s in stems] == ["s1", None, "s1"]
    assert all(np.array_equal(s.placement.position, stems[1].placement.position) for s in stems)
