#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, ammonyte as amt
from ammonyte.utils.ks import KS_test
ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
transitions = KS_test(ngrip, w_min=0.12, w_max=2.5, n_w=15, d_c=0.77, n_c=3, s_c=2.0, x_c=0.8)
print(f"Detected {len(transitions[0])} transitions")


# In[2]:


import os, ammonyte as amt
from ammonyte.utils.lerm_transitions import lerm_transition

# Load data and perform LERM analysis
ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
NGRIP_td = amt.TimeEmbeddedSeries(ngrip, m=11)
NGRIP_epsilon = NGRIP_td.find_epsilon(eps=1, target_density=0.05)
NGRIP_rm = NGRIP_epsilon['Output']
NGRIP_lp = NGRIP_rm.laplacian_eigenmaps(w_size=20, w_incre=4)
NGRIP_lp_smooth = amt.utils.fisher.smooth_series(NGRIP_lp, block_size=3)

# Detect transitions
jump_times, jump_directions, upper_bound, lower_bound = lerm_transition(NGRIP_lp_smooth)
print(f"Detected {len(jump_times)} transitions")
print(f"Bounds used: upper={upper_bound:.4f}, lower={lower_bound:.4f}")


# In[3]:


from ammonyte.utils.metrics import evaluate_detection

detected = [10, 25, 50, 75]
ground_truth = [10, 26, 60]

metrics = evaluate_detection(detected, ground_truth, tolerance=2)
print(metrics)


# In[4]:


from ammonyte.utils.metrics import evaluate_detection

detected = [9.8, 23.5, 45.2]
ground_truth = [10, 25, 40, 60]

metrics = evaluate_detection(detected, ground_truth, tolerance=1.0)

print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")
print(f"F1 Score: {metrics.f1_score:.3f}")


# In[5]:


import os, ammonyte as amt
from ammonyte.utils.ruptures_transitions import ruptures_transition
ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
transitions = ngrip.ruptures(algo='Pelt', cost='rbf', pen=5)
print(f"Detected {len(transitions.jump_times)} transitions")

