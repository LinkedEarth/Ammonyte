#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, ammonyte as amt

# Load data
ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))

# LERM analysis
NGRIP_td = amt.TimeEmbeddedSeries(ngrip, m=11)
NGRIP_epsilon = NGRIP_td.find_epsilon(eps=1, target_density=0.05)
NGRIP_rm = NGRIP_epsilon['Output']
NGRIP_lp = NGRIP_rm.laplacian_eigenmaps(w_size=20, w_incre=4)
NGRIP_lp_smooth = amt.utils.fisher.smooth_series(NGRIP_lp, block_size=3)

# Detect transitions
transitions = NGRIP_lp_smooth.lerm_transitions()
print(transitions)
transitions.plot()


# In[2]:


transitions = NGRIP_lp_smooth.lerm_transitions(
    transition_interval=(0.025, 0.010)
)


# In[3]:


import os, ammonyte as amt
ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
transitions = ngrip.kstest(w_min=0.12, w_max=2.5, n_w=15, d_c=0.77, n_c=3, s_c=2, x_c=0.8)
print(transitions)


# In[4]:


print(f"D-statistics: {transitions.d_statistics}")
print(f"P-values: {transitions.p_values}")


# In[5]:


transitions.plot()


# In[6]:


import os, ammonyte as amt
ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
transitions = ngrip.ruptures(algo='Pelt', cost='rbf', pen=5)
print(transitions)


# In[7]:


transitions.plot()


# In[8]:


import os, ammonyte as amt
ngrip = amt.Series.from_csv(os.path.join(os.path.dirname(amt.__file__), 'data', 'NGRIP.csv'))
transitions = ngrip.kstest(w_min=0.12, w_max=2.5, n_w=15, d_c=0.77, n_c=3, s_c=2, x_c=0.8)
print(transitions)


# In[9]:


transitions.plot()


# In[10]:


print(f"Number of transitions: {len(transitions.jump_times)}")
print(f"First transition: {transitions.jump_times[0]:.2f}")

