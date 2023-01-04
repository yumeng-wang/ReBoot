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
    return(x1 - x2)
  }else{
    return(x1 + x2)
  }
}

gradient <- function(X, y, beta){
  n <- nrow(X)
  sum <- rep(0, ncol(X))
  for(i in 1:n){
    sum <- sum + c((t(X[i,]) %*% beta) ^ 2 - y[i]) * c(t(X[i,]) %*% beta) * X[i,] 
  }
  return(sum / n)
}

hessian <- function(X, y, beta){
  n <- nrow(X)
  sum <- matrix(0, ncol(X), ncol(X))
  for(i in 1:n){
    sum <- sum + c(3 * (t(X[i,]) %*% beta) ^ 2 - y[i]) * X[i,] %*% t(X[i,])
  }
  return(sum / n)
}

wirtinger_flow <- function(X, y, beta = FALSE, lr = 5e-3, times = 10000){
  n <- nrow(X)
  if(!beta[1]){
    S <- matrix(0, ncol(X), ncol(X))
    for(i in 1:n){
      S <- S + y[i] * X[i,] %*% t(X[i,]) / n
    }
    beta <- sqrt(eigen(S)$values[1]/3) * eigen(S)$vectors[,1]
  }
  for(i in 1:times){
    grad <- gradient(X, y, beta)
    beta <- beta - lr * grad
    if(l2norm(grad) <= 1e-6) break
  }
  return(beta)
}

#-----------------------------------------------
p <- 5
R <- 10
N <- 12000
mnum <- c(300, 250, 200, 150, 100, 50)

iters <- 200
beta_true <- rep(1, p)

all_estimator <- list()
len_run <- rep(0, length(mnum))

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

for(s in 1:length(mnum)){
  m <- mnum[s]
  n <- N / m
  nR <- n * R
  NR <- N * R
  cat("the number of machine is ", m, "\n", sep="")
  cat("the number of local sample size is ", n, "\n", sep="")
  
  est_list <- foreach(t = 1:iters, .packages = c("MASS")) %dopar% {
    res <- tryCatch({
      set.seed(t)
      ##----------------generate data----------------
      X <- matrix(rnorm(N * p),  N, p)
      y <- (X %*% beta_true) ^ 2 + rnorm(N) 
      
      ##--------------optimal estimator--------------
      est_opt <- wirtinger_flow(X, y)
      
      ##--------------average estimator--------------
      est_local <- matrix(0, m, p)
      for(j in 1:m){
        lst <- seq(((j - 1) * n + 1), (j * n))
        est_local[j,] <- wirtinger_flow(X[lst,], y[lst])
        if((j >= 2) & (sign(est_local[j,1]) != sign(est_local[1,1]))){
          est_local[j,] <- - est_local[j,]
        }
      }
      est_aver <- colMeans(est_local)
      
      ##---------------CSL1 estimator----------------
      lgrad1 <- gradient(X, y, est_aver)
      hess1 <- hessian(X[1:n,], y[1:n], est_aver)
      est_csl1 <- est_aver - solve(hess1) %*% lgrad1
      
      #---------------csl2 estimator---------------
      lgrad2 <- gradient(X, y, est_csl1)
      hess2 <- hessian(X[1:n,], y[1:n], est_csl1)
      est_csl2 <- est_csl1 - solve(hess2) %*% lgrad2
      rm(X); rm(y)
      
      ##--------------reboot estimator---------------
      Z <- matrix(rnorm(NR * p),  NR, p)
      yz <- unlist(lapply(1:m, function(j) 
        (Z[((j - 1) * nR + 1):(j * nR),] %*% est_local[j,]) ^ 2 + rnorm(nR)))
      est_reboot <- wirtinger_flow(Z, yz, est_local[1,], lr = 5e-2)
      
      #--------------return results-------------------
      list(opt=est_opt, aver=est_aver, reboot=est_reboot, local=est_local,
           csl1=est_csl1, csl2=est_csl2)
    },error = function(e){
      return(NA)
    })  
      
    #--------------report results-------------------
    cat(s, ":", t, " done\n", sep="")
    return(res)
  }
  est_list <- est_list[!is.na(est_list)]
  all_estimator[[s]] <- est_list
  len_run[s] <- length(est_list)
  
  est_opt_all <- sapply(est_list, function(res) res$opt)
  est_ave_all <- sapply(est_list, function(res) res$ave)
  est_reboot_all <- sapply(est_list, function(res) res$reboot)
  est_csl1_all <- sapply(est_list, function(res) res$csl1)
  est_csl2_all <- sapply(est_list, function(res) res$csl2)
  
  mse_opt_all <- sapply(1:length(est_list), function(j) mse(est_opt_all[,j], beta_true))
  mse_ave_all <- sapply(1:length(est_list), function(j) mse(est_ave_all[,j], beta_true))
  mse_reboot_all <- sapply(1:length(est_list), function(j) mse(est_reboot_all[,j], beta_true))
  mse_csl1_all <- sapply(1:length(est_list), function(j) mse(est_csl1_all[,j], beta_true))
  mse_csl2_all <- sapply(1:length(est_list), function(j) mse(est_csl2_all[,j], beta_true))
  
  # adjust the sign
  for(j in 2:length(est_list)){
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
save(all_estimator, file = "est_all_5.RData")
save(df_mse, df_bias, mnum, N, file = "est_res_5.RData")

df_mse
df_bias
len_run
