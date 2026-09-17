import config


def compute_threshold(detector):
    loo_scores = detector.score_leave_one_out()
    worst_name, worst_score = max(loo_scores, key=lambda t: t[1])
    threshold = worst_score * config.THRESHOLD_MARGIN
    return threshold, loo_scores, worst_name