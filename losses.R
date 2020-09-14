df<-read.csv('training.txt')

plot(df$epoch,df$loss,type="o", col="blue", pch="o", lty=1,xlab="Epoch",ylab="Loss")
points(df$epoch,df$val_loss, col="red", pch="*")
lines(df$epoch,df$val_loss, col="red",lty=2)
legend(x='topleft', legend=c("Loss", "Validation Loss"),
       col=c("blue","red"), lty=1:2, cex=0.8)

plot(df$epoch,df$sparse_categorical_accuracy,type="o", col="blue", pch="o", lty=1,xlab="Epoch",ylab="Accuracy")
points(df$epoch,df$val_sparse_categorical_accuracy, col="red", pch="*")
lines(df$epoch,df$val_sparse_categorical_accuracy, col="red",lty=2)
legend(x='bottomright', legend=c("Accuracy", "Validation Accuracy"),
       col=c("blue","red"), lty=1:2, cex=0.8)
