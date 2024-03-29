library(MASS)
library(doSNOW)
library(foreach)
cluster <- makeCluster(11, outfile="")
registerDoSNOW(cluster)

gradient <- function(X, y ,w){
    sig <- 1 / (1 + exp(- X %*% w))
    return(t(X) %*% (y - sig))
}

hessian <- function(X ,w){
    sig <- 1 / (1 + exp( - X %*% w))
    gma <- diag(c(sig)) %*% diag(c(1 - sig))
    return( - t(X) %*% gma %*% X)
}

logistic_coef <- function(X, y){
    fit <- glm(y ~ X - 1, family = binomial, control = list(maxit = 500))
    coef_fit <- coef(fit)
    return(as.vector(coef_fit))
}

l2norm <- function(x){
    return(sum(x ^ 2))
}

#-----------------------------------------
d <- 10
R <- 100  # reboot sample size rate
N <- 12000
mnum <- c(120, 100, 80, 60, 40, 20)

rou <- 0.5
meanx <- rep(0, d)
Sigmax <- matrix(0, d, d) 
Ix <- diag(1, d, d)
for (i in 1:d){
    for (j in 1:d){
        Sigmax[i,j] <- rou^abs(i - j)
    }
}
cat("p is ", d, "\n", sep="")

iters <- 200
theta <- rep(.2, d)

est_all <- list()
mse_opt_mean <- rep(0, length(mnum))
mse_aver_mean <- rep(0, length(mnum))
mse_csl1_mean <- rep(0, length(mnum))
mse_csl2_mean <- rep(0, length(mnum))
mse_reboot_I_mean <- rep(0, length(mnum))
mse_reboot_S_mean <- rep(0, length(mnum))

mse_opt_sd <- rep(0, length(mnum))
mse_aver_sd <- rep(0, length(mnum))
mse_csl1_sd <- rep(0, length(mnum))
mse_csl2_sd <- rep(0, length(mnum))
mse_reboot_I_sd <- rep(0, length(mnum))
mse_reboot_S_sd <- rep(0, length(mnum))

bias_opt <- rep(0, length(mnum))
bias_aver <- rep(0, length(mnum))
bias_csl1 <- rep(0, length(mnum))
bias_csl2 <- rep(0, length(mnum))
bias_reboot_I <- rep(0, length(mnum))
bias_reboot_S <- rep(0, length(mnum))

for(s in 1:length(mnum)){
    m <- mnum[s]
    n <- N / m
    nR <- n * R
    NR <- N * R
    cat("the number of machine is ", m, "\n", sep="")
    cat("the number of local sample size is ", n, "\n", sep="")
    
    est_list <- foreach(i = 1:iters, .packages = c("MASS")) %dopar% {
        set.seed(i)
        #---------------generating data---------------
        X <- mvrnorm(N, meanx, Sigmax) 
        p <- 1 / (1 + exp(- X %*% theta))
        y <- rbinom(n = N, size = 1, p = p)
        rm(X1); rm(X2); rm(X3)
        
        #--------------optimal estimator--------------
        est_opt <- logistic_coef(X, y)
        
        #--------------average estimator--------------
        est_local <- sapply(1:m, function(j){
            lst <- seq((j - 1) * n + 1,  j * n)
            coef <- logistic_coef(X[lst,], y[lst])
            return(coef)})
        est_aver <- rowMeans(est_local)
        
        #---------------csl1 estimator---------------
        lgrad1_all <- sapply(1:m, function(j){
            lst <- seq((j - 1) * n + 1,  j * n)
            grad <- gradient(X[lst,], y[lst], est_aver)
            return(grad)})
        lgrad1 <- rowMeans(lgrad1_all)
        hess1 <- hessian(X[1:n,], est_aver)
        est_csl1 <- est_aver - solve(hess1) %*% lgrad1
        
        #---------------csl2 estimator---------------
        lgrad2_all <- sapply(1:m, function(j){
            lst <- seq((j - 1) * n + 1,  j * n)
            grad <- gradient(X[lst,], y[lst], est_csl1)
            return(grad)})
        lgrad2 <- rowMeans(lgrad2_all)
        hess2 <- hessian(X[1:n,], est_csl1)
        est_csl2 <- est_csl1 - solve(hess2) %*% lgrad2
        rm(X); rm(p); rm(y)
        
        #---------------reboot estimator I----------------
        Z <- mvrnorm(NR, meanx, Ix)
        pz <- lapply(1:m, function(j) 
            1 / (1 + exp(- Z[((j - 1) * nR + 1) : (j * nR),] %*% est_local[, j])))
        pz <- as.vector(unlist(pz))
        yz <- rbinom(n = NR, size = 1, p = pz)
        est_reboot_I <- logistic_coef(Z, yz)
        rm(Z); rm(pz); rm(yz)
        
        #---------------reboot estimator S----------------
        Z <- mvrnorm(NR, meanx, Sigmax)
        pz <- lapply(1:m, function(j) 
            1 / (1 + exp(- Z[((j - 1) * nR + 1) : (j * nR),] %*% est_local[, j])))
        pz <- as.vector(unlist(pz))
        yz <- rbinom(n = NR, size = 1, p = pz)
        est_reboot_S <- logistic_coef(Z, yz)
        rm(Z); rm(pz); rm(yz)
        
        #--------------report results-------------------
        cat(s, ":", i, " done\n", sep="")
        return(list(opt=est_opt, ave=est_aver, csl1=est_csl1, csl2=est_csl2, reboot_I=est_reboot_I, reboot_S=est_reboot_S))
    }
    est_all[[s]] <- est_list
    
    est_opt_all <- sapply(est_list, function(res) res$opt)
    est_ave_all <- sapply(est_list, function(res) res$ave)
    est_csl1_all <- sapply(est_list, function(res) res$csl1)
    est_csl2_all <- sapply(est_list, function(res) res$csl2)
    est_reboot_I_all <- sapply(est_list, function(res) res$reboot_I)
    est_reboot_S_all <- sapply(est_list, function(res) res$reboot_S)
    
    mse_opt_all <- sapply(1:iters, function(j) l2norm(est_opt_all[,j] - theta))
    mse_ave_all <- sapply(1:iters, function(j) l2norm(est_ave_all[,j] - theta))
    mse_csl1_all <- sapply(1:iters, function(j) l2norm(est_csl1_all[,j] - theta))
    mse_csl2_all <- sapply(1:iters, function(j) l2norm(est_csl2_all[,j] - theta))
    mse_reboot_I_all <- sapply(1:iters, function(j) l2norm(est_reboot_I_all[,j] - theta))
    mse_reboot_S_all <- sapply(1:iters, function(j) l2norm(est_reboot_S_all[,j] - theta))
    
    mse_opt_mean[s] <- mean(mse_opt_all)
    mse_aver_mean[s] <- mean(mse_ave_all)
    mse_csl1_mean[s] <- mean(mse_csl1_all)
    mse_csl2_mean[s] <- mean(mse_csl2_all)
    mse_reboot_I_mean[s] <- mean(mse_reboot_I_all)
    mse_reboot_S_mean[s] <- mean(mse_reboot_S_all)
    
    mse_opt_sd[s] <- sd(mse_opt_all)
    mse_aver_sd[s] <- sd(mse_ave_all)
    mse_csl1_sd[s] <- sd(mse_csl1_all)
    mse_csl2_sd[s] <- sd(mse_csl2_all)
    mse_reboot_I_sd[s] <- sd(mse_reboot_I_mean)
    mse_reboot_S_sd[s] <- sd(mse_reboot_S_mean)
    
    bias_opt[s] <- l2norm(rowMeans(est_opt_all) - theta)
    bias_aver[s] <-  l2norm(rowMeans(est_ave_all) - theta)
    bias_csl1[s] <-  l2norm(rowMeans(est_csl1_all) - theta)
    bias_csl2[s] <-  l2norm(rowMeans(est_csl2_all) - theta)
    bias_reboot_I[s] <-  l2norm(rowMeans(est_reboot_I_all) - theta)
    bias_reboot_S[s] <-  l2norm(rowMeans(est_reboot_S_all) - theta)
}

save(est_all, file="est_all_rou_0.5.RData")
df_mse_mean <- data.frame(mse_opt_mean, mse_aver_mean, mse_csl1_mean, mse_csl2_mean, mse_reboot_I_mean, mse_reboot_S_mean)
df_mse_sd <- data.frame(mse_opt_sd, mse_aver_sd, mse_csl1_sd, mse_csl2_sd, mse_reboot_I_sd, mse_reboot_S_sd)
df_bias <- data.frame(bias_opt, bias_aver, bias_csl1, bias_csl2, bias_reboot_I, bias_reboot_S)

save(df_mse_mean, df_mse_sd, df_bias, mnum, N, file="est_res_rou_0.5.RData", col.names=T)
df_mse_mean
df_mse_sd
df_bias
