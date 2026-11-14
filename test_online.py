import online as OL


def test_new_listing_starts_at_the_model_and_learns_from_results(tmp_path):
    o = OL.Online(str(tmp_path / 'online.json'))
    t = 1_000_000.0
    assert o.offset('A1', t) == (0.0, 0.0)
    # Nick at $20.00, model says 20%; snapshots show 60%; his price changes 2 h later -> one result, gap logit(.6)-logit(.2)
    o.on_decision('A1', 20.00, 0.2, t)
    for k in range(6):
        o.on_snapshot('A1', 0.6, t + 1200 * (k + 1))
    o.on_decision('A1', 19.99, 0.3, t + 7200)
    off, n = o.offset('A1', t + 7200)
    gap = OL.logit(0.6) - OL.logit(0.2)
    assert n == 1.0 and abs(off - gap / (1 + OL.K)) < 1e-9
    p, _, _ = o.corrected('A1', 0.2, t + 7200)
    assert 0.2 < p < 0.6                                    # moved toward the listing's record, not all the way


def test_windows_end_after_6_hours_and_evidence_fades(tmp_path):
    o = OL.Online(None)
    t = 0.0
    o.on_decision('A1', 10.0, 0.5, t)
    o.on_snapshot('A1', 1.0, t + 600)
    o.on_snapshot('A1', 1.0, t + 6 * 3600 + 1)              # past 6 h: closes the window, this snapshot isn't counted
    assert o.closed == 1 and 'A1' not in o.win
    off1, n1 = o.offset('A1', t + 6 * 3600 + 1)
    off2, n2 = o.offset('A1', t + 6 * 3600 + 1 + 12 * 3600)  # one half-life later
    assert abs(n2 - n1 / 2) < 1e-9 and 0 < off2 < off1
    o.on_decision('A2', 10.0, 0.5, t); o.on_decision('A2', 9.0, 0.5, t + 60)   # no snapshot in the window: no result
    assert o.offset('A2', t + 60) == (0.0, 0.0)


def test_shadow_logs_the_online_correction_without_using_it(tmp_path):
    import shadow as S
    from test_shadow import MODEL
    sh = S.Shadow(str(tmp_path), lambda *a: None, lambda: '2026-11-14T00:00:00Z')
    sh.online.ev['A1'] = [10.0, 10.0, __import__('time').time()]    # a listing that ran 1 log-odds above the model
    body = {'featuredBuyingOptions': [{'buyingOptionType': 'New', 'segmentedFeaturedOffers': [
        {'sellerId': S.ME, 'featuredOfferSegments': [{'segmentDetails': {'glanceViewWeightPercentage': 75}}]},
        {'sellerId': 'X', 'featuredOfferSegments': [{'segmentDetails': {'glanceViewWeightPercentage': 25}}]}]}]}
    sh.online.on_decision('A1', 30.0, 0.3)
    sh.on_competitive_summary('A1', body)
    assert sh.online.win['A1'][3] == [0.75]
    p, off, n = sh.online.corrected('A1', 0.3)
    assert abs(off - 10.0 / 30.0) < 1e-3 and p > 0.3
