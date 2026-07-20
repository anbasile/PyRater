// pyrater note: this is Paun, Carpenter, Chamberlain, Hovy, Kruschwitz &
// Poesio (2018), "Comparing Bayesian Models of Annotation" (TACL, CC-BY
// 4.0)'s Logistic Random Effects (non-centered) supplementary-material
// implementation, not original to this library. Migrated from Stan's
// pre-2.26 array syntax (e.g. `int y[N]`) to the modern `array[N] int y`
// syntax required by current CmdStan releases; purely syntactic, no change
// in model semantics.

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
  vector[K] alpha = rep_vector(1,K); //class prevalence prior
}

parameters {
  simplex[K] pi;

  matrix[I,K-1] theta_raw;
  matrix<lower=0>[K,K-1] Chi;

  array[J] matrix[K,K-1] beta_raw;
  matrix[K,K-1] zeta;
  matrix<lower=0>[K,K-1] Omega;
}

transformed parameters {

  array[I] vector[K] log_q_c;
  vector[K] log_pi;
  array[J] matrix[K,K-1] beta;
  array[I] matrix[K,K-1] theta;

  log_pi = log(pi);

  for(j in 1:J)
    beta[j] = zeta + Omega .* beta_raw[j]; //non-centered parameterization

  for (i in 1:I)
  {
    for(h in 1:K)
    {
      log_q_c[i,h] = log_pi[h];

      theta[i,h] = Chi[h] .* theta_raw[i]; //non-centered parameterization
    }

  }

  for (n in 1:N)
  {
    for (h in 1:K)
    {
      row_vector[K-1] temp = beta[jj[n],h] - theta[ii[n],h];
      row_vector[K] temp_i = append_col(temp, 0); //fix the last category to 0

      log_q_c[ii[n],h] = log_q_c[ii[n],h] + temp_i[y[n]] - log_sum_exp(temp_i);
    }
  }
}

model
{
  pi ~ dirichlet(alpha);

  to_vector(zeta) ~ normal(0, 1);
  to_vector(Omega) ~ normal(0, 1);

  to_vector(Chi) ~ normal(0, 1);

  for(j in 1:J)
    to_vector(beta_raw[j]) ~ normal(0, 1);

  for (i in 1:I)
  {
    theta_raw[i] ~ normal(0, 1);
    target += log_sum_exp(log_q_c[i]);
  }
}

generated quantities {
  // NOTE: this generated quantities block is a pyrater addition, not part of
  // the original paper's supplementary material. It is purely additive
  // (does not touch `parameters`/`model`, so posterior samples are
  // bit-for-bit unchanged) and only materializes normalized, per-class
  // confusion-matrix-like quantities -- beta_norm (annotator ability) and
  // theta_norm (item difficulty) -- the same way HDS-NC.stan and ID-NC.stan
  // already do as transformed parameters, so pyrater can read them back for
  // model-native credibility/difficulty estimates without reimplementing
  // Stan's own softmax transform by hand in Python.
  array[I] vector[K] q_z; //the true class distribution of each item
  array[J] matrix[K,K] beta_norm; //log-softmax-normalized annotator confusion matrix
  array[I] matrix[K,K] theta_norm; //log-softmax-normalized item difficulty profile
  vector[K] zeros = rep_vector(0,K);

  for(i in 1:I)
    q_z[i] = softmax(log_q_c[i]);

  for(j in 1:J)
  {
    beta_norm[j] = append_col(beta[j], zeros);
    for(h in 1:K)
      beta_norm[j,h] = beta_norm[j,h] - log_sum_exp(beta_norm[j,h]);
  }

  for(i in 1:I)
  {
    theta_norm[i] = append_col(theta[i], zeros);
    for(h in 1:K)
      theta_norm[i,h] = theta_norm[i,h] - log_sum_exp(theta_norm[i,h]);
  }
}
