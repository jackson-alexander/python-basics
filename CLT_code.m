%% Jackson Alexander - CLT
% Central Limit Theorem Program 08/06/2023

% Settings
clc;
clear;
close all;
cd("/Users/jacksonalexander/Desktop/TA Stuff/Metrics 1/Recitation 1/")

%% Simulation

rng(0)

n=[1 2 10 100];
samples=1000;
a=0;b=1;

Xn=a+(b-a)*rand(max(n),samples); % U(a,b) - draw samples samples of max(n) obs iid
mu=(b-a)/2;

CLT = zeros(length(n),samples); % preallocate for speed
for i=1:length(n)
    Xn_temp=Xn(1:n(i),:); % picks first n iid rows (all cols) of Xn matrix
    Xn_temp_bar = mean(Xn_temp,1); % mean of each column/sample
    CLT(i,:) = sqrt(n(i))*(Xn_temp_bar-mu); % 1-by-samples -d-> N(0,sigma^2)
end

% Animated Figures (beginners may want to opt for normal figures):
figure(1)
t = tiledlayout('flow');
for j=0:10:samples
    for i=1:length(n)
        nexttile(i)
        histogram(CLT(i,1:j),'BinEdges',(-(b+mu/2):((b+mu/2)/15):b+mu/2))
        xlim([-(b+mu/2) b+mu/2])
        ylim([0 150])
        lgd=legend(sprintf('n=%d',n(i)));
        fontsize(lgd,12,"points")
        drawnow limitrate;
    end
end

