// STAN code for naive Model 3

data{
    int<lower = 1> N; // number of balloons

    int<lower = 0> pumps[N]; // number of pumps in each balloon
    int<lower=0, upper=1> popped[N]; // whether the balloons are popped
    

    int<lower = 1> i_max; // number of maximum pumps ever tried

}

parameters{
    real<lower = 0.0> gamma;
    real<lower = 0.0> a0;
    real<lower = 0.0> m0;
    real<lower = 0.0> beta;
}

model{
    real a;
    real m;
    real q;
    real g;

    for (i in 1:N){
        q = a/(a+m);
        g = -gamma/log(q);
        for (j in 1:pumps[i]){
            1 ~ bernoulli(1/(1+exp(beta*(i-g))));
        }
        a += pumps[i];
        m += popped[i];
    }
}