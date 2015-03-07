
import os

from acculturation.experiments import corpcorp
from acculturation.experiments.corpcorp.member_distances import get_userlevel_distances

# Directory path to pickled or json files.
DIRNAME = None

######################################################################
# User-level experiments, no monthly breakdown:

# The central experiment, I'd say: LIWC mapping, all people in the data, all words no breakdown by time:
users_liwc = get_userlevel_distances(DIRNAME, sampling=False, liwc_map=True)

# Basic plot with stat test:
corpcorp.plots.plot_js_distances(users_liwc)

# Sampling functions:
def get_min_length(data):
    return min([len(data['living']), len(data['departed'])])

def get_half_of_min_length(data):
    return int(round(get_min_length(data) / 2.0, 0))

# Sample from the larger class:
corpcorp.plots.plot_js_distances(users_liwc, sampsize=get_min_length(users_liwc))

# Sample from both classes:
corpcorp.plots.plot_js_distances(users_liwc, sampsize=get_half_of_min_length(users_liwc))

# Variation on the user-level experiment: no LIWC mapping, sample 1000 vocab items, keep all users:
users_vocab1000 = get_userlevel_distances(DIRNAME, sampling=False, liwc_map=False, vocabsize=1000)

# Variation on the user-level experiment: LIWC mapping, all words, sample 1000 users
users_liwc_interlocutors1000 = get_userlevel_distances(DIRNAME, sampling=True, sampsize=1000, liwc_map=True)

######################################################################
# Monthly breakdown

# The central temporal experiment: LIWC mapping, all people in the data, distances taken by month,
# minimum of 20 messages per user per month
months_liwc = get_userlevel_distances(DIRNAME, monthly=True, min_segment_size=20, liwc_map=True)

# Variation on the temporal experiment: no LIWC mapping, all people in the data, distances taken by month:
months = get_userlevel_distances(DIRNAME, monthly=True, min_segment_size=20, liwc_map=False)

def get_monthy_means(monthly):
    monthly_means = defaultdict(dict)
    for key in ('living', 'departed'):
        for u, date_dict in monthly[key].items(): 
            if date_dict: # Keep only users for whom we have at least one monthly estimate.
                monthly_means[key][u] = np.mean(np.array(date_dict.values()))
    return monthly_means

# Plot monthly averages:
corpcorp.plots.plot_js_distances(
    get_monthy_means(months_liwc),
    title="Mean monthly linguistic distance from one's email interlocutors",
    ylabel="Mean monthly Jensen-Shannon distance")

# Try plotting trajectories:
corpcorp.plots.plot_trajectories(
    months_liwc['departed'],
    min_months=4,
    max_months=12,
    zscore=False,
    logscale=True)
