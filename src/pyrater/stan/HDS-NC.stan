// pyrater note: this is Paun, Carpenter, Chamberlain, Hovy, Kruschwitz &
// Poesio (2018), "Comparing Bayesian Models of Annotation" (TACL, CC-BY
// 4.0)'s Hierarchical Dawid & Skene (non-centered) supplementary-material
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

transformed data
{
  vector[K] alpha = rep_vector(1,K); //class prevalence prior
  vector[K] zeros = rep_vector(0,K);
}

parameters {
  simplex[K] pi;
  array[J] matrix[K,K-1] beta_raw;
  matrix[K,K-1] zeta;
  matrix<lower=0>[K,K-1] Omega;
}

transformed parameters
{
  array[J] matrix[K,K-1] beta;
  array[J] matrix[K,K] beta_norm;
  array[I] vector[K] log_q_c;
  vector[K] log_pi;

  for(j in 1:J)
  {
    beta[j] = zeta + Omega .* beta_raw[j]; //non centered parameterization
    beta_norm[j] = append_col(beta[j], zeros); //fix last category to 0 (softmax non-identifiability)
    for(h in 1:K)
      beta_norm[j,h] = beta_norm[j,h] - log_sum_exp(beta_norm[j,h]); //log_softmax
  }

  log_pi = log(pi);
  for (i in 1:I)
    log_q_c[i] = log_pi;

  for (n in 1:N)
   for (h in 1:K)
     log_q_c[ii[n],h] = log_q_c[ii[n],h] + beta_norm[jj[n],h,y[n]];
}

model {

  pi ~ dirichlet(alpha);
  to_vector(zeta) ~ normal(0,1);
  to_vector(Omega) ~ normal(0,1);

  for(j in 1:J)
    to_vector(beta_raw[j]) ~ normal(0, 1); //part of the non centered parameterization

  for (i in 1:I)
    target += log_sum_exp(log_q_c[i]);
}

generated quantities {
  array[I] vector[K] q_z; //the true class distribution of each item

  for(i in 1:I)
    q_z[i] = softmax(log_q_c[i]);
}
