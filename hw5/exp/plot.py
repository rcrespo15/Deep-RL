import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn
    pythonw plot.py data/ac_PM_bc0_s8_PointMass-v0_13-11-2018_20-43-05 AverageReturn
    pythonw plot.py data/ac_PM_hist_bc0.01_s8_PointMass-v0_11-11-2018_19-09-28 AverageReturn
    pythonw plot.py data/ac_HC_bc0.0001_kl0.1_dlr0.005_dti10000_HalfCheetah-v2_13-11-2018_11-16-14 AverageReturn
    pythonw plot.py data/ac_HC_bc0.001_kl0.1_dlr0.005_dti1000_HalfCheetah-v2_13-11-2018_10-17-15 AverageReturn
    pythonw plot.py data/ac_PM_ex2_s8_bc0.05_kl0.1_dlr0.001_dh8_dti1000_PointMass-v0_13-11-2018_09-09-41 AverageReturn

    Question 1:
    pythonw plot.py data/ac_PM_hist_bc0.01_s8_PointMass-v0_14-11-2018_15-20-51 data/ac_PM_bc0_s8_PointMass-v0_13-11-2018_20-43-05 AverageReturn

    Question 2:
    pythonw plot.py data/ac_PM_rbf_bc0.01_s8_sig0.2_PointMass-v0_13-11-2018_22-10-01 data/ac_PM_bc0_s8_PointMass-v0_11-11-2018_17-43-10 AverageReturn

    Question 3:
    pythonw plot.py data/ac_PM_ex2_s8_bc0.05_kl0.1_dlr0.001_dh8_dti1000_PointMass-v0_14-11-2018_15-45-36 data/ac_PM_bc0_s8_PointMass-v0_11-11-2018_17-43-10 AverageReturn


    Question 4:
    pythonw plot.py data/ac_HC_bc0_HalfCheetah-v2_14-11-2018_15-51-18 AverageReturn
    pythonw plot.py data/ac_HC_bc0.001_kl0.1_dlr0.005_dti1000_HalfCheetah-v2_14-11-2018_16-32-22 AverageReturn
    pythonw plot.py data/ac_HC_bc0.0001_kl0.1_dlr0.005_dti10000_HalfCheetah-v2_14-11-2018_17-27-25 AverageReturn
    pythonw plot.py data/ac_HC_bc0_HalfCheetah-v2_13-11-2018_09-28-17 data/ac_HC_bc0.001_kl0.1_dlr0.005_dti1000_HalfCheetah-v2_13-11-2018_10-17-15 data/ac_HC_bc0.0001_kl0.1_dlr0.005_dti10000_HalfCheetah-v2_13-11-2018_11-16-14 AverageReturn
and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data, value="AverageReturn"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition")
    plt.legend(loc='best').draggable()
    # plt.legend(loc='best', bbox_to_anchor=(1, 1), fontsize=8).draggable()
    plt.show()


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root,'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']

            log_path = os.path.join(root,'log.txt')
            experiment_data = pd.read_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
                )

            datasets.append(experiment_data)
            unit += 1

    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    for value in values:
        plot_data(data, value=value)

if __name__ == "__main__":
    main()
