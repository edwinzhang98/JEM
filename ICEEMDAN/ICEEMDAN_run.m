% Example of the iCEEMDAN performance, used in the work where iCEEMDAN was first presented:
Nstd = 0.2;
NR = 100;
MaxIter = 5000;


[modes its]=iceemdan(gzt,0.2,100,5000);
t=1:length(gzt);

[a b]=size(modes);

figure;
subplot(a+1,1,1);
plot(t,gzt);% the gzt signal is in the first row of the subplot
ylabel('gzt')
set(gca,'xtick',[])
axis tight;

for i=2:a
    subplot(a+1,1,i);
    plot(t,modes(i-1,:));
    ylabel (['IMF ' num2str(i-1)]);
    set(gca,'xtick',[])
    xlim([1 length(gzt)])
end;

subplot(a+1,1,a+1)
plot(t,modes(a,:))
ylabel(['IMF ' num2str(a)])
xlim([1 length(gzt)])

figure;
boxplot(its);