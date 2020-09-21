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

plot_stats<-function(name,x,y,y_val,ylab,xlegend){
  plot(x, y,
       type = "o",
       col  = "blue",
       pch  = "o",
       lty  = 1,
       ylim = c(min(y,y_val),max(y,y_val)),
       xlab = "Epoch",
       ylab = ylab)
  
  points(x,y_val,
         col = "red",
         pch = "*")
  
  lines(x,y_val,
        col = "red",
        lty = 2)
  
  legend(x      = xlegend, 
         legend = c(paste("Training", ylab),
                    paste("Validation", ylab)),
         col    = c("blue","red"),
         lty    = 1:2,
         cex    = 0.8)
  
  title(name)
  grid(col="darkgrey")
}



plot_results<-function(name) {
  df<-read.csv(name)
  plot_stats(name,rownames(df),
             df$loss,df$val_loss,
             "Loss",'topright')
  plot_stats(name,rownames(df),
             df$sparse_categorical_accuracy,
             df$val_sparse_categorical_accuracy,
             "Accuracy",'bottomright')
}

plot_results('flowers.txt')

