# Copyright (C) 2020 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

# Plot loss and accuracy for Training and Validation data
# from logfiles

setwd("~/../learn")
rm(list=ls())
cat("\014") 
if(!is.null(dev.list())) dev.off()

plot.metric<-function(file.name,
                     x,y,y.validation,
                     metric.name,x.legend){
  plot(x, y,
       type = "o",
       col  = "blue",
       pch  = "o",
       lty  = 1,
       ylim = c(min(y,y.validation),
                max(y,y.validation)),
       xlab = "Epoch",
       ylab = metric.name)
  
  points(x,y.validation,
         col = "red",
         pch = "*")
  
  lines(x,y.validation,
        col = "red",
        lty = 2)
  
  legend(x      = x.legend, 
         legend = c(paste("Training", metric.name),
                    paste("Validation", metric.name)),
         col    = c("blue","red"),
         lty    = 1:2,
         cex    = 0.8)
  
  title(file.name)
  grid(col="darkgrey")
}



plot.metrics<-function(file.name) {
  df<-read.csv(file.name)
  plot.metric(file.name,rownames(df),
             df$loss,df$val_loss,
             "Loss",'topright')
  plot.metric(file.name,rownames(df),
             df$sparse_categorical_accuracy,
             df$val_sparse_categorical_accuracy,
             "Accuracy",'bottomright')
}

plot.metrics('flowers.txt')


