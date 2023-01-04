library(MASS)
library(doSNOW)
library(foreach)
cluster <- makeCluster(35, outfile="")
registerDoSNOW(cluster)

gradient <- function(X, y ,w){
  return(-t(X) %*% (y - exp(X %*% w)) / nrow(X))
}

hessian <- function(X ,w){
  gma <- diag(c(exp(X %*% w)))
  return(t(X) %*% gma %*% X / nrow(X))
}

possion_coef <- function(X, y){
  fit <- glm(y ~ X - 1, family = "poisson", control = list(maxit = 500, epsilon = 1e-14))
  coef_fit <- coef(fit)
  return(as.vector(coef_fit))
}

l2norm <- function(x){
  return(sum(x ^ 2))
}

#-----------------------------------------
d <- 5
R <- 100  # reboot sample size rate
r <- 0.5  # subsampling rate
N <- 12000
mnum <- c(600, 500, 400, 300, 200, 100)
cat("p is ", d, "\n", sep="")

iters <- 200
theta <- rep(0.1, d)

est_all <- list()
mse_opt_mean <- rep(0, length(mnum))
mse_aver_mean <- rep(0, length(mnum))
mse_savgm_mean <- rep(0, length(mnum))
mse_reboot_mean <- rep(0, length(mnum))
mse_csl1_mean <- rep(0, length(mnum))
mse_csl2_mean <- rep(0, length(mnum))

mse_opt_sd <- rep(0, length(mnum))
mse_aver_sd <- rep(0, length(mnum))
mse_savgm_sd <- rep(0, length(mnum))
mse_reboot_sd <- rep(0, length(mnum))
mse_csl1_sd <- rep(0, length(mnum))
mse_csl2_sd <- rep(0, length(mnum))

bias_opt <- rep(0, length(mnum))
bias_aver <- rep(0, length(mnum))
bias_savgm <- rep(0, length(mnum))
bias_reboot <- rep(0, length(mnum))
bias_csl1 <- rep(0, length(mnum))
bias_csl2 <- rep(0, length(mnum))

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
    X <- matrix(c(runif(N), runif(N * (d - 1), -1, 1)), N, d)
    y <- rpois(n = N, lambda = exp(X %*% theta))
    
    #--------------optimal estimator--------------
    est_opt <- possion_coef(X, y)
    
    #--------------average estimator--------------
    est_local <- sapply(1:m, function(j){
      lst <- seq((j - 1) * n + 1,  j * n)
      coef <- possion_coef(X[lst,], y[lst])
      return(coef)})
    est_aver <- rowMeans(est_local)
    
    #--------------savgm estimator----------------
    est_local2 <- sapply(1:m, function(j){
      lst <- seq((j - 1) * n + 1,  (j - 1) * n + ceiling(r * n))
      coef <- possion_coef(X[lst,], y[lst])
      return(coef)})
    est_aver2 <- rowMeans(est_local2)
    est_savgm <- (est_aver - r * est_aver2) / (1 - r)
    
    #---------------csl1 estimator---------------
    lgrad1 <- gradient(X, y, est_aver)
    hess1 <- hessian(X[1:n,], est_aver)
    est_csl1 <- est_aver - solve(hess1) %*% lgrad1
    
    #---------------csl2 estimator---------------
    lgrad2 <- gradient(X, y, est_csl1)
    hess2 <- hessian(X[1:n,], est_csl1)
    est_csl2 <- est_csl1 - solve(hess2) %*% lgrad2
    rm(X); rm(y)
    
    #---------------reboot estimator----------------
    Z <- matrix(c(runif(NR), runif(NR * (d - 1), -1, 1)), NR, d)
    lambdaz <- unlist(lapply(1:m, function(j) exp(Z[((j - 1) * nR + 1) : (j * nR),] %*% est_local[, j])))
    yz <- rpois(n = NR, lambda = lambdaz)
    est_reboot <- possion_coef(Z, yz)
    rm(Z); rm(yz); rm(lambdaz)
    
    #--------------report results-------------------
    cat(s, ":", i, " done\n", sep="")
    return(list(opt=est_opt, ave=est_aver, savgm=est_savgm, reboot=est_reboot,
                csl1=est_csl1, csl2=est_csl2))
  }
  est_all[[s]] <- est_list
  
  est_opt_all <- sapply(est_list, function(res) res$opt)
  est_ave_all <- sapply(est_list, function(res) res$ave)
  est_savgm_all <- sapply(est_list, function(res) res$savgm)
  est_reboot_all <- sapply(est_list, function(res) res$reboot)
  est_csl1_all <- sapply(est_list, function(res) res$csl1)
  est_csl2_all <- sapply(est_list, function(res) res$csl2)
  
  mse_opt_all <- sapply(1:iters, function(j) l2norm(est_opt_all[,j] - theta))
  mse_ave_all <- sapply(1:iters, function(j) l2norm(est_ave_all[,j] - theta))
  mse_savgm_all <- sapply(1:iters, function(j) l2norm(est_savgm_all[,j] - theta))
  mse_reboot_all <- sapply(1:iters, function(j) l2norm(est_reboot_all[,j] - theta))
  mse_csl1_all <- sapply(1:iters, function(j) l2norm(est_csl1_all[,j] - theta))
  mse_csl2_all <- sapply(1:iters, function(j) l2norm(est_csl2_all[,j] - theta))
  
  mse_opt_mean[s] <- mean(mse_opt_all)
  mse_aver_mean[s] <- mean(mse_ave_all)
  mse_savgm_mean[s] <- mean(mse_savgm_all)
  mse_reboot_mean[s] <- mean(mse_reboot_all)
  mse_csl1_mean[s] <- mean(mse_csl1_all)
  mse_csl2_mean[s] <- mean(mse_csl2_all)
  
  mse_opt_sd[s] <- sd(mse_opt_all)
  mse_aver_sd[s] <- sd(mse_ave_all)
  mse_savgm_sd[s] <- sd(mse_savgm_all)
  mse_reboot_sd[s] <- sd(mse_reboot_all)
  mse_csl1_sd[s] <- sd(mse_csl1_all)
  mse_csl2_sd[s] <- sd(mse_csl2_all)
  
  bias_opt[s] <- l2norm(rowMeans(est_opt_all) - theta)
  bias_aver[s] <-  l2norm(rowMeans(est_ave_all) - theta)
  bias_savgm[s] <-  l2norm(rowMeans(est_savgm_all) - theta)
  bias_reboot[s] <-  l2norm(rowMeans(est_reboot_all) - theta)
  bias_csl1[s] <-  l2norm(rowMeans(est_csl1_all) - theta)
  bias_csl2[s] <-  l2norm(rowMeans(est_csl2_all) - theta)
}

save(est_all, file="est_all_5.RData")
df_mse_mean <- data.frame(mse_opt_mean, mse_aver_mean, mse_savgm_mean, mse_reboot_mean,
                          mse_csl1_mean, mse_csl2_mean)
df_mse_sd <- data.frame(mse_opt_sd, mse_aver_sd, mse_savgm_sd, mse_reboot_sd, mse_csl1_sd, mse_csl2_sd)
df_bias <- data.frame(bias_opt, bias_aver, bias_savgm, bias_reboot, bias_csl1, bias_csl2)

save(df_mse_mean, df_mse_sd, df_bias, mnum, N, file="est_res_5.RData", col.names=T)

df_mse_mean
df_mse_sd
df_bias
