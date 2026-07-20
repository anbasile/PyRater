// pyrater note: this is Paun, Carpenter, Chamberlain, Hovy, Kruschwitz &
// Poesio (2018), "Comparing Bayesian Models of Annotation" (TACL, CC-BY
// 4.0)'s MACE supplementary-material implementation, not original to this
// library. Migrated from Stan's pre-2.26 array syntax (e.g. `int y[N]`) to
// the modern `array[N] int y` syntax required by current CmdStan releases;
// purely syntactic, no change in model semantics.

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
  vector[K] alpha = rep_vector(1.0/K,K);
  vector[K] eta = rep_vector(10,K);
}

parameters {
  array[J] simplex[K] epsilon;
  array[J] real<lower=0, upper=1> theta;
}

transformed parameters {
  array[I] vector[K] log_q_c;
  vector[K] log_alpha;

  log_alpha = log(alpha);

  for (i in 1:I)
    log_q_c[i] = log_alpha;

  for (n in 1:N)
  {
    for (h in 1:K)
    {
      int indicator = (y[n] == h);
      log_q_c[ii[n],h] = log_q_c[ii[n],h] + log( theta[jj[n]] * indicator + (1-theta[jj[n]]) * epsilon[jj[n],y[n]] );
    }
  }
}

model {
  for(j in 1:J)
  {
    epsilon[j] ~ dirichlet(eta);
    theta[j] ~ beta(0.5,0.5);
  }

  for (i in 1:I)
    target += log_sum_exp(log_q_c[i]);
}

generated quantities {
  array[I] vector[K] q_z; //the true class distribution of each item

  for(i in 1:I)
    q_z[i] = softmax(log_q_c[i]);
}
