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

setwd("~/../learn")
rm(list=ls())
cat("\014") 
if(!is.null(dev.list())) dev.off()

plot_results<-function(name) {
  df<-read.csv(name)
  
  plot(rownames(df),
       df$loss,
       type="o",
       col="blue",
       pch="o",
       lty=1,
       xlab="Epoch",
       ylab="Loss")
  points(rownames(df),
         df$val_loss,
         col="red",
         pch="*")
  lines(rownames(df),
        df$val_loss,
        col="red",
        lty=2)
  legend(x='topleft', 
         legend=c("Loss", "Validation Loss"),
         col=c("blue","red"),
         lty=1:2, cex=0.8)
  title(name)
  
  plot(rownames(df),
       df$sparse_categorical_accuracy,
       type="o",
       col="blue",
       pch="o",
       lty=1,
       xlab="Epoch",
       ylab="Accuracy")
  points(rownames(df),
         df$val_sparse_categorical_accuracy,
         col="red",
         pch="*")
  lines(rownames(df),
        df$val_sparse_categorical_accuracy,
        col="red",
        lty=2)
  legend(x='bottomright',
         legend=c("Accuracy", "Validation Accuracy"),
         col=c("blue","red"),
         lty=1:2,
         cex=0.8)
  title(name)
  
}

plot_results('flowers.txt')

