import matplotlib
matplotlib.use("Agg")


def visualization_smoke_test(analysis):
    analysis.plot_analysis_results(
        frame_num=0,
        detected_params={'plot': True},
        matched_params={'plot': True, 'legend': True},
        plot_result=False,
    )
