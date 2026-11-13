import json, time, datetime as dt
import history as HM


def body(asin, t, offers):
    return json.dumps({'Payload': {'AnyOfferChangedNotification': {'OfferChangeTrigger': {'ASIN': asin, 'TimeOfOfferChange': t},
        'Offers': [{'SellerId': s, 'SubCondition': 'new', 'ListingPrice': {'Amount': p}, 'Shipping': {'Amount': sh}, 'IsFulfilledByAmazon': f,
                    'IsBuyBoxWinner': w, **({'ShippingTime': {'AvailabilityType': 'FUTURE_WITH_DATE'}} if bo else {})}
                   for s, p, sh, f, w, bo in offers]}}})


def iso(sec_ago):
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=sec_ago)).strftime('%Y-%m-%dT%H:%M:%SZ')


def test_history_keeps_change_points_and_counts_changes(tmp_path):
    h = HM.History(str(tmp_path))
    h.add([body('A1', iso(7200), [('ME', 30.0, 0, True, True, False), ('R1', 31.0, 0, True, False, False), ('F1', 28.0, 4.99, False, False, False)]),
           body('A1', iso(7000), [('ME', 30.0, 0, True, True, False), ('R1', 31.0, 0, True, False, False), ('F1', 28.0, 4.99, False, False, False)]),  # no change
           body('A1', iso(3600), [('ME', 29.5, 0, True, False, False), ('R1', 29.49, 0, True, True, True), ('F1', 28.0, 4.99, False, False, False)]),
           body('A2', iso(3600), [('X', 10.0, 0, True, True, False)]), 'not json'])
    q = h.query('A1', hours=3, me='ME')
    assert len(q['points']) == 2                                         # the repeat is dropped
    assert q['points'][1][1]['R1'] == [29.49, 1] and q['points'][1][2] == 'R1'   # backordered flag, Buy Box winner
    assert q['sellers']['F1']['fba'] is False and q['sellers']['F1']['low'] == 32.99   # price + shipping
    assert q['sellers']['ME']['me'] and q['sellers']['ME']['changes'] == 1 and q['sellers']['R1']['changes'] == 1
    q = h.query('A1', hours=1.5, me='ME')                                # the state before the window starts the lines
    assert q['points'][0][0] == q['from'] and q['points'][0][1]['ME'] == [30.0, 0]
    h.prune(); assert h.stats['rows'] == 4


def test_held_buy_box_since(tmp_path):
    h = HM.History(str(tmp_path))
    h.add([body('A1', iso(7200), [('ME', 30.0, 0, True, True, False), ('R1', 31.0, 0, True, False, False)]),
           body('A1', iso(3600), [('ME', 30.0, 0, True, False, False), ('R1', 29.0, 0, True, True, False)])])
    now = time.time()
    assert h.held('A1', now - 1800, 'ME') is False        # lost it an hour ago, nothing since
    assert h.held('A1', now - 5400, 'ME') is True         # held it at the start of a 90-minute window
    h.add([body('A1', iso(600), [('ME', 28.99, 0, True, True, False), ('R1', 29.0, 0, True, False, False)])])
    assert h.held('A1', now - 1800, 'ME') is True
