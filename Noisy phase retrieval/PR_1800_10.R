library(MASS)
library(doSNOW)
library(foreach)
cluster <- makeCluster(35, outfile = "")
registerDoSNOW(cluster)

l2norm <- function(x){
  return(sum(x ^ 2))
}

mse <- function(x1, x2){
  return(min(l2norm(x1 - x2), l2norm(x1 + x2)))
}

bias <- function(x1, x2){
  if(l2norm(x1 - x2) <= l2norm(x1 + x2)){
    bias_x <- x1 - x2
  }else{
    bias_x <- x1 + x2
  }
  return(bias_x)
}

gradient <- function(X, y, beta){
  n <- nrow(X)
  p <- ncol(X)
  sum <- rep(0, p)
  for(i in 1:n){
    sum <- sum + c((t(X[i,]) %*% beta) ^ 2 - y[i]) * c(t(X[i,]) %*% beta) * X[i,] 
  }
  return(sum / n)
}

hessian <- function(X, y, beta){
  n <- nrow(X)
  p <- ncol(X)
  sum <- matrix(0, p, p)
  for(i in 1:n){
    sum <- sum + c(3 * (t(X[i,]) %*% beta) ^ 2 - y[i]) * X[i,] %*% t(X[i,])
  }
  return(sum / n)
}

wirtinger_flow <- function(X, y, init = FALSE, lr = 0.001, times = 1000){
  n <- nrow(X)
  p <- ncol(X)
  
  if(!init){
    S <- matrix(0, p, p)
    for(i in 1:n){
      S <- S + y[i] * X[i,] %*% t(X[i,]) / n
    }
    eigen_Y <- eigen(S)
    beta <- sqrt(eigen_Y$values[1]/3) * eigen_Y$vectors[,1]
  }else{
    beta <- init
  }
  
  for(i in 1:times){
    beta <- beta - lr * gradient(X, y, beta)
  }
  return(beta)
}

#-----------------------------------------------
p <- 10
R <- 10
N <- 1800
mnum <- c(10, 15, 20, 25, 30, 36)

iters <- 200
beta_true <- rep(1, p)

mse_opt_mean <- vector(length = length(mnum))
mse_aver_mean <- vector(length = length(mnum))
mse_reboot_mean <- vector(length = length(mnum))
mse_csl1_mean <- rep(0, length(mnum))
mse_csl2_mean <- rep(0, length(mnum))

bias_opt <- rep(0, length(mnum))
bias_aver <- rep(0, length(mnum))
bias_reboot <- rep(0, length(mnum))
bias_csl1 <- rep(0, length(mnum))
bias_csl2 <- rep(0, length(mnum))

all_estimator <- list()

for(s in 1:length(mnum)){
  m <- mnum[s]
  n <- N / m
  nR <- n * R
  NR <- N * R
  cat("the number of machine is ", m, "\n", sep="")
  cat("the number of local sample size is ", n, "\n", sep="")
  
  est_list <- foreach(t = 1:iters, .packages = c("MASS")) %dopar% {
    set.seed(t)
    ##----------------generate data----------------
    X <- matrix(rnorm(N * p),  N, p)
    y <- (X %*% beta_true) ^ 2 + rnorm(N) 
    
    ##--------------optimal estimator--------------
    est_opt <- wirtinger_flow(X, y)
    
    ##--------------average estimator--------------
    est_local <- matrix(0, m, p)
    for(i in 1:m){
      est_local[i,] <- wirtinger_flow(X[((i-1) * n + 1):(i * n),], y[((i-1) * n + 1):(i * n)])
    }
    for(i in 2:m){
      if(sign(est_local[i,1]) != sign(est_local[1,1])){
        est_local[i,] <- - est_local[i,]
      }
    }
    est_aver <- colMeans(est_local)
    
    ##---------------CSL1 estimator----------------
    lgrad1_all <- sapply(1:m, function(i){
      lst <- seq((i - 1) * n + 1,  i * n)
      grad <- gradient(X[lst,], y[lst], est_aver)
      return(grad)})
    lgrad1 <- rowMeans(lgrad1_all)
    hess1 <- hessian(X[1:n,], y[1:n], est_aver)
    est_csl1 <- est_aver - solve(hess1) %*% lgrad1
    
    #---------------csl2 estimator---------------
    lgrad2_all <- sapply(1:m, function(i){
      lst <- seq((i - 1) * n + 1,  i * n)
      grad <- gradient(X[lst,], y[lst], est_csl1)
      return(grad)})
    lgrad2 <- rowMeans(lgrad2_all)
    hess2 <- hessian(X[1:n,], y[1:n], est_csl1)
    est_csl2 <- est_csl1 - solve(hess2) %*% lgrad2
    rm(X); rm(y)
    
    ##--------------reboot estimator---------------
    X_reboot <- matrix(rnorm(NR * p),  NR, p)
    y_reboot <- lapply(1:m, function(i){ 
      set.seed(i * 1000 + t)
      (X_reboot[((i-1) * nR + 1):(i * nR),] %*% est_local[i,]) ^ 2 + rnorm(nR)})
    est_reboot <- wirtinger_flow(X_reboot, unlist(y_reboot), est_local[1,], lr = 0.01)
    
    ##--------------report results------------------
    cat(s, ":", t, " done\n", sep="")
    return(list(opt=est_opt, aver=est_aver, reboot=est_reboot, local=est_local,
                csl1=est_csl1, csl2=est_csl2))
  }
  all_estimator[[s]] <- est_list
  
  est_opt_all <- sapply(est_list, function(res) res$opt)
  est_ave_all <- sapply(est_list, function(res) res$ave)
  est_reboot_all <- sapply(est_list, function(res) res$reboot)
  est_csl1_all <- sapply(est_list, function(res) res$csl1)
  est_csl2_all <- sapply(est_list, function(res) res$csl2)
  
  mse_opt_all <- sapply(1:iters, function(j) mse(est_opt_all[,j], beta_true))
  mse_ave_all <- sapply(1:iters, function(j) mse(est_ave_all[,j], beta_true))
  mse_reboot_all <- sapply(1:iters, function(j) mse(est_reboot_all[,j], beta_true))
  mse_csl1_all <- sapply(1:iters, function(j) mse(est_csl1_all[,j], beta_true))
  mse_csl2_all <- sapply(1:iters, function(j) mse(est_csl2_all[,j], beta_true))
  
  # adjust the sign
  for(j in 2:iters){
    if(sign(est_opt_all[1,j]) != sign(est_opt_all[1,1])){
      est_opt_all[,j] <- - est_opt_all[,j]
    }
    if(sign(est_ave_all[1,j]) != sign(est_ave_all[1,1])){
      est_ave_all[,j] <- - est_ave_all[,j]
    }
    if(sign(est_reboot_all[1,j]) != sign(est_reboot_all[1,1])){
      est_reboot_all[,j] <- - est_reboot_all[,j]
    }
    if(sign(est_csl1_all[1,j]) != sign(est_csl1_all[1,1])){
      est_csl1_all[,j] <- - est_csl1_all[,j]
    }
    if(sign(est_csl2_all[1,j]) != sign(est_csl2_all[1,1])){
      est_csl2_all[,j] <- - est_csl2_all[,j]
    }
  }
  bias_opt[s] <- mse(rowMeans(est_opt_all), beta_true)
  bias_aver[s] <-  mse(rowMeans(est_ave_all), beta_true)
  bias_reboot[s] <-  mse(rowMeans(est_reboot_all), beta_true)
  bias_csl1[s] <-  mse(rowMeans(est_csl1_all), beta_true)
  bias_csl2[s] <-  mse(rowMeans(est_csl2_all), beta_true)
  
  mse_opt_mean[s] <- mean(mse_opt_all)
  mse_aver_mean[s] <- mean(mse_ave_all)
  mse_reboot_mean[s] <- mean(mse_reboot_all)
  mse_csl1_mean[s] <- mean(mse_csl1_all)
  mse_csl2_mean[s] <- mean(mse_csl2_all)
  
  print(mse_opt_mean)
  print(mse_aver_mean)
  print(mse_reboot_mean)
  print(mse_csl1_mean)
  print(mse_csl2_mean)
  
  print(bias_opt)
  print(bias_aver)
  print(bias_reboot)
  print(bias_csl1)
  print(bias_csl2)
}


df_mse <- data.frame(mse_opt_mean, mse_aver_mean, mse_reboot_mean, mse_csl1_mean, mse_csl2_mean)
df_bias <- data.frame(bias_opt, bias_aver, bias_reboot, bias_csl1, bias_csl2)
save(all_estimator, file = "est_all_1800_10.RData")
save(df_mse, df_bias, mnum, N, file = "est_res_1800_10.RData")

df_mse
df_bias
