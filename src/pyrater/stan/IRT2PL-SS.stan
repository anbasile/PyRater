// pyrater note: this is Han, Adiguzel & Carpenter (2026), "Hierarchical
// Bayesian Crowdsourcing with Item Difficulty" (arXiv:2405.19521), reduction
// "a" of their Table 1 -- IRT-2PL with per-annotator sensitivity and
// specificity, the model that won on both of their datasets.  Adapted from
// stan_hierarchical/a.stan of https://github.com/seongwoohan/bayes-crowdsourcing-irt
// (MIT, Copyright (c) 2024 Bob Carpenter).
//
// Changes from the original, all of them interface rather than model:
//   - the `#include` files are flattened into this one, since pyrater loads a
//     single .stan file per model;
//   - the data block takes pyrater's contract (J, K, N, I, ii, jj, y with y in
//     1..K) instead of the original's (item, rater, rating in 0..1), and
//     converts between them;
//   - `lambda` is dropped rather than set to a zero vector -- reduction "a" IS
//     lambda = 0, so multiplying by (1 - lambda) is a no-op kept only so the
//     original could share one file across reductions;
//   - `log_q_c`, `q_z` and a pointwise `log_lik` are exposed under the names
//     the rest of pyrater expects;
//   - the posterior predictive check reports pyrater's per-class vote
//     histogram rather than the original's vote_count/rater_count pair.
//
// WHEN TO REACH FOR THIS MODEL RATHER THAN THE OTHERS.
//
// Two reasons, and the second decides whether the model is usable at all in a
// given analysis.
//
// First, it separates an item's DIFFICULTY from its DISCRIMINATION.  An item
// everyone gets wrong and an item only good annotators get right are different
// objects; Dawid & Skene distinguishes neither, and ID-NC and LRE-NC capture
// the first but not the second.  The paper's own finding is that Dawid & Skene
// "fails to capture the ratings distribution adequately" -- it "inflates the
// number of middling positive votes compared to the data" -- while this model
// does not.  That is a posterior predictive statement, invisible to R-hat and
// ESS; the rep_vote_hist block below is what surfaces it.
//
// Second: IT IS ORIENTATION-IDENTIFIED WITHOUT GOLD.  Every symmetric
// annotation model -- Dawid & Skene, hierarchical D&S, item difficulty,
// multinomial -- has a latent class labelling that Stan cannot pin down.
// `align_chains` fixes the WITHIN-fit ambiguity by matching chains to each
// other; only `align_to_gold` fixes the GLOBAL one, and it needs gold labels.
// An analysis that has none can therefore come back with the classes
// systematically swapped: every reported quantity is its own mirror image,
// rank correlations against a known quality signal arrive with the sign
// reversed, and accuracy arrives near its own complement -- with clean
// convergence diagnostics throughout, because the chains agree with each other
// perfectly about the labelling they all chose.
//
// The constraint below rules that out.  Writing
//
//     logit_alpha_sens = -logit_alpha_spec + log1p_exp(eta)
//
// forces logit_sens + logit_spec > 0: every annotator is better than chance.
// Under a global flip sens' = 1 - spec and spec' = 1 - sens, so the sum lands
// below chance and the flipped mode falls outside the support.  The labelling
// is identified by the model rather than by the data.
//
// THE COST IS AN ASSUMPTION, and it is a real restriction.  Annotators must be
// better than chance.  That is safe for trained annotators, for crowdworkers
// after spam filtering, and for an ensemble of reasonable classifiers.  It is
// NOT safe where genuinely adversarial annotators may be present -- a pool
// under attack, or one where a systematic misunderstanding inverts someone's
// labels.  Use a symmetric model there, and resolve the orientation some other
// way.

data {
  int<lower=1> J; //number of annotators
  int<lower=2> K; //number of classes
  int<lower=1> N; //number of annotations
  int<lower=1> I; //number of items
  array[N] int<lower=1,upper=I> ii; //the item the n-th annotation belongs to
  array[N] int<lower=1,upper=J> jj; //the annotator which produced the n-th annotation
  array[N] int y; //the class of the n-th annotation
}

transformed data {
  // Sensitivity and specificity are binary notions; the model does not
  // generalise to K > 2 and must say so rather than fit something meaningless.
  if (K != 2) {
    reject("irt_2pl_ss is a binary model (sensitivity/specificity); got K = ", K);
  }

  // pyrater passes classes as 1..K; the original is written on 0/1 ratings.
  // Class 2 is "positive", so sensitivity is the rate of reporting class 2 on
  // an item whose true class is 2.  Which real label that is depends on the
  // order `AnnotationData` factorized them in, so the two annotator
  // parameters are only interpretable against `class_labels` -- the FIT
  // does not depend on it.
  array[N] int<lower=0,upper=1> rating;
  for (n in 1:N) {
    rating[n] = y[n] - 1;
  }

  // Sizing for the posterior predictive check.
  array[I] int n_ratings_ppc = rep_array(0, I);
  for (n in 1:N) {
    n_ratings_ppc[ii[n]] += 1;
  }
  int R_max_ppc = max(n_ratings_ppc);
}

parameters {
  real<lower=0, upper=1> pi;      //prevalence of the positive class
  vector[2] mu_alpha;             //population mean of (eta, logit specificity)
  vector<lower=0>[2] tau_alpha;   //population scale
  cholesky_factor_corr[2] L_Omega_alpha;
  matrix[2, J] z_alpha;           //non-centred annotator effects
  vector[I] beta;                 //item difficulty
  vector<lower=0>[I] delta;       //item discrimination
}

transformed parameters {
  matrix[2, J] alpha_latent;
  vector[J] eta_alpha;
  vector[J] logit_alpha_spec;
  vector[J] logit_alpha_sens;
  array[I] vector[K] log_q_c;

  alpha_latent = rep_matrix(mu_alpha, J)
                 + diag_pre_multiply(tau_alpha, L_Omega_alpha) * z_alpha;

  for (j in 1:J) {
    eta_alpha[j] = alpha_latent[1, j];
    logit_alpha_spec[j] = alpha_latent[2, j];
    // The identifiability constraint. See the header: this is what makes the
    // model usable without gold.
    logit_alpha_sens[j] = -logit_alpha_spec[j] + log1p_exp(eta_alpha[j]);
  }

  // log_q_c[i, 1] is the negative class, log_q_c[i, 2] the positive one, in
  // the same layout every other pyrater model uses.
  for (i in 1:I) {
    log_q_c[i, 1] = log1m(pi);
    log_q_c[i, 2] = log(pi);
  }

  for (n in 1:N) {
    int i = ii[n];
    int j = jj[n];
    log_q_c[i, 2] += bernoulli_lpmf(
      rating[n] | inv_logit(delta[i] * (logit_alpha_sens[j] - beta[i])));
    log_q_c[i, 1] += bernoulli_lpmf(
      rating[n] | inv_logit(-delta[i] * (logit_alpha_spec[j] - beta[i])));
  }
}

model {
  pi ~ beta(2, 2);
  mu_alpha ~ normal(0, 5);
  tau_alpha ~ lognormal(0, 1);
  L_Omega_alpha ~ lkj_corr_cholesky(2);
  to_vector(z_alpha) ~ std_normal();
  beta ~ normal(0, 1);
  delta ~ lognormal(0, 0.25);

  for (i in 1:I) {
    target += log_sum_exp(log_q_c[i]);
  }
}

generated quantities {
  array[I] vector[K] q_z; //the true class distribution of each item

  // Pointwise over ITEMS, as everywhere else in pyrater: the latent class is
  // marginalised, so the likelihood factorises over items rather than over the
  // individual ratings that make one up.
  vector[I] log_lik;

  // Posterior predictive check: for each class, the histogram over items of
  // how many votes that class received.
  array[I] int<lower=1,upper=K> z_rep;
  array[K, R_max_ppc + 1] int rep_vote_hist = rep_array(0, K, R_max_ppc + 1);

  for (i in 1:I) {
    q_z[i] = softmax(log_q_c[i]);
    log_lik[i] = log_sum_exp(log_q_c[i]);
    z_rep[i] = categorical_rng(q_z[i]);
  }

  {
    array[I, K] int rep_counts = rep_array(0, I, K);
    for (n in 1:N) {
      int i = ii[n];
      int j = jj[n];
      // The emission the likelihood scores the observed rating with: an item's
      // discrimination times the gap between the annotator's ability on the true
      // class and the item's difficulty.
      real p = z_rep[i] == 2
                 ? inv_logit(delta[i] * (logit_alpha_sens[j] - beta[i]))
                 : inv_logit(-delta[i] * (logit_alpha_spec[j] - beta[i]));
      int y_rep_n = bernoulli_rng(p) + 1;
      rep_counts[i, y_rep_n] += 1;
    }
    for (i in 1:I)
      for (k in 1:K)
        rep_vote_hist[k, rep_counts[i, k] + 1] += 1;
  }
}
