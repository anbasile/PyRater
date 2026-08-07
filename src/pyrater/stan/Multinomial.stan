// pyrater note: this is Paun, Carpenter, Chamberlain, Hovy, Kruschwitz &
// Poesio (2018), "Comparing Bayesian Models of Annotation" (TACL, CC-BY
// 4.0)'s Multinomial supplementary-material implementation, not original to
// this library. Migrated from Stan's pre-2.26 array syntax (e.g.
// `int y[N]`) to the modern `array[N] int y` syntax required by current
// CmdStan releases; purely syntactic, no change in model semantics.

data {
  // int<lower=1> J; //number of annotators
  int<lower=2> K; //number of classes
  int<lower=1> N; //number of annotations
  int<lower=1> I; //number of items
  array[N] int<lower=1,upper=I> ii; //the item the n-th annotation belongs to
  // int<lower=1,upper=J> jj[N]; //the annotator which produced the n-th annotation
  array[N] int y; //the class of the n-th annotation
}

transformed data {
  vector[K] alpha = rep_vector(1,K); //class prevalence prior
  vector[K] eta = rep_vector(1,K); //class abilities prior

  // pyrater addition: sizing for the posterior predictive check below.  Only
  // the maximum is needed -- the observed histogram is recomputed from the
  // data in Python, because Stan does not write transformed data to the CSV.
  array[I] int n_ratings_ppc = rep_array(0, I);
  for (n in 1:N)
    n_ratings_ppc[ii[n]] += 1;
  int R_max_ppc = max(n_ratings_ppc);
}

parameters {
  simplex[K] pi;
  array[K] simplex[K] zeta;
}

transformed parameters {
  array[I] vector[K] log_q_c;
  vector[K] log_pi;
  array[K] vector[K] log_zeta;

  log_pi = log(pi);
  log_zeta = log(zeta);

  for (i in 1:I)
    log_q_c[i] = log_pi;

  for (n in 1:N)
    for (h in 1:K)
      log_q_c[ii[n],h] = log_q_c[ii[n],h] + log_zeta[h,y[n]];
}

model {
  for(h in 1:K)
    zeta[h] ~ dirichlet(eta);

  pi ~ dirichlet(alpha);

  for (i in 1:I)
    target += log_sum_exp(log_q_c[i]);
}

generated quantities {
  array[I] vector[K] q_z; //the true class distribution of each item

  // pyrater addition: pointwise log-likelihood for LOO/WAIC via ArviZ.
  //
  // THE POINTWISE UNIT IS THE ITEM, not the annotation.  These models
  // marginalize the latent class z_i, so the likelihood factorizes over items
  // and not over the individual ratings that make one up -- `model` above
  // accumulates exactly `log_sum_exp(log_q_c[i])` per item.  So a comparison
  // built on this is leave-one-ITEM-out: it asks how well a model predicts a
  // whole rating vector, which is the right question when the models differ in
  // how they describe raters and items.
  //
  // `parameters` and `model` are untouched, so the posterior DISTRIBUTION
  // is unchanged.  The realised draws are not: the `_rng` calls below share
  // Stan's base RNG with the sampler's momentum draws, so a fit run before
  // this addition cannot be reproduced bit-for-bit after it.
  vector[I] log_lik;

  // pyrater addition: posterior predictive check.
  //
  // WHY, WHEN R-hat AND ESS ALREADY PASS.  Convergence diagnostics say the
  // sampler explored the posterior of the model it was given; they say nothing
  // about whether that model can produce data like the observed data.  Han,
  // Adiguzel & Carpenter (arXiv:2405.19521) show Dawid & Skene passing
  // convergence while failing exactly here -- it "inflates the number of
  // middling positive votes compared to the data".
  //
  // THE STATISTIC.  For each class k, the histogram over items of how many
  // votes that class received: rep_vote_hist[k, c + 1] counts the items where
  // class k drew exactly c replicated votes.  Generalises the paper's check to
  // K > 2 and reduces to it at K = 2.  Per-item counts stay in a local block so
  // the CSV carries K*(R+1) numbers per draw rather than I*K.
  array[I] int<lower=1,upper=K> z_rep;
  array[K, R_max_ppc + 1] int rep_vote_hist = rep_array(0, K, R_max_ppc + 1);

  for(i in 1:I) {
    q_z[i] = softmax(log_q_c[i]);
    log_lik[i] = log_sum_exp(log_q_c[i]);
    z_rep[i] = categorical_rng(q_z[i]);
  }

  {
    array[I, K] int rep_counts = rep_array(0, I, K);
    for (n in 1:N) {
      // No annotator term at all: one emission shared by everyone.
      // This model is in the ladder to show what the rater terms buy.
      int y_rep_n = categorical_rng(zeta[z_rep[ii[n]]]);
      rep_counts[ii[n], y_rep_n] += 1;
    }
    for (i in 1:I)
      for (k in 1:K)
        rep_vote_hist[k, rep_counts[i, k] + 1] += 1;
  }
}
