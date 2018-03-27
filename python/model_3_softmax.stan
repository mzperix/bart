// STAN code for naive Model 3
functions{
    real marginal_log_probability(real a, real m, int i){
        return(lgamma(a+i)-lgamma(a)+lgamma(a+m)-lgamma(a+m+i));
    }

    int argmax(vector v, int i_max){
        int c;
        real m;
        c = 1;
        m = v[1];
        for (i in 2:i_max){
            if (m<v[i]){
                c = i;
                m = v[i];
            }
        }
        return(c);
    }
}

data{
    int<lower = 1> N; // number of balloons

    int<lower = 0> pumps[N]; // number of pumps in each balloon
    int<lower = 0, upper=1> popped[N]; // whether the balloons are popped

    int<lower = 1> i_max; // number of maximum pumps ever tried
    vector[i_max] rewards;

    real<lower = 0.0> gamma_std;
    real<lower = 0.0> a0_std;
    real<lower = 0.0> m0_std;
    real<lower = 0.0> beta_std;
}

parameters{
    real<lower = 0.0> gamma_pos;
    real<lower = 1.0> a0;
    real<lower = 1.0> m0;
    real<lower = 0.0> beta_soft_max;
}

model{
    real a;
    real m;
    vector[i_max] expected_utilities;
    real p;

    // PRIORS
    a0 ~ normal(0, a0_std);
    m0 ~ normal(0, m0_std);

    beta_soft_max ~ normal(0,beta_std);
    gamma_pos ~ normal(0,gamma_std);

    a = a0;
    m = m0;
    for (i in 1:N){
        for (j in 1:i_max){
            expected_utilities[j] = (rewards[j]^gamma_pos)*exp(marginal_log_probability(a, m, j-1));
        }
        
        if (popped[i] == 0)        
            pumps[i] ~ categorical_logit(expected_utilities*beta_soft_max);
                       // equiv. to categorical(softmax(.))

        if (popped[i] == 1){
            p = sum(softmax(expected_utilities*beta_soft_max)[pumps[i]:]);
            if (p<1)
                1 ~ bernoulli(p);
        }

        a += pumps[i]-1-popped[i];
        m += popped[i];
    }
}
