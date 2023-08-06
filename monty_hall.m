%% Jackson Alexander - Monty Hall
% 08/06/2023

% Settings
clc;
clear;
close all;
cd("/Users/jacksonalexander/Desktop/TA Stuff/Metrics 1/Recitation 1/")

%% Simulation

rng(pi) % for reproducability

num_doors = 3; % number of doors available

samples = 3000; % number of samples

repititions = 250; % number of repititions in each sample

prize = randi(num_doors,repititions,samples); % repititions-by-samples
% chooses number [1,num_doors] (picks door with prize) 
% for every possible sample-repitition pair
% now we do the same but for the participant choosing a door:
select = randi(num_doors,repititions,samples); % repititions-by-samples
% note: the easiest way to do this is to realize that if we are making the
% participant switch doors every time, they will loose if prize=select.
% So, you could just code lose = (prize==select) to get a binary matrix,
% where 1 is lose and 0 is win.
% But, in the spirit of the game and for fun, we will code every step:

swap = zeros(repititions,samples); % preallocate the variable we want to store
for i = 1:samples % loop over index of samples
    for j = 1:repititions % loop over index of repititions
        % we don't need to store the reveal variable:
        reveal = 1:num_doors; % row vector of all doors
        % note: prize and select match the column index for reveal
        reveal([prize(j,i) select(j,i)]) = []; % remove price and select doors
        % Now, Monty will always reveal the first num_doors-2 doors (WLOG)
        % this really only matters for prize=select case
        reveal=reveal(1:num_doors-2);
        % for example: prize=select=1, num_doors=100, reveal doors 2-99
        % now we can swap to the door that's leftover:
        swap_temp = 1:num_doors;
        swap_temp([select(j,i) reveal]) = []; % can't swap to these doors
        
        % now we want to store the door we swap to:
        swap(j,i) = swap_temp; % scalar
    end
end

% now, we want to see the win percentage for each sample
% we win only if swap=prize:
win = (prize==swap); % repititions-by-samples
% 1 if win, 0 if lose for each possible sample-repitition pair
win_percent = mean(win); % 1-by-samples
% now we just take the mean of win for each sample
% this gets is the % of wins since win is binary

% we also want to see what is happening to the cumulative win %
% across trials (trials = samples * repititions)
win_percent2 = zeros(1,samples); % preallocate
for i = 1:samples % loop over index of samples
    win_percent2(i) = mean(win(:,1:i),"all"); % mean over all dimensiona
    % cumulative win % at each sample
end
% now, we're done and can plot the win percentage

% Animated Figures (beginners may want to opt for normal figures):
figure(1)
t=tiledlayout(2,1);
title(t,'Simulated Win % When Door Switched')
for i=1:samples/100:samples
    nexttile(1)
    plot(1:i,win_percent2(1:i))
    ylim([min(win_percent2) max(win_percent2)])
    xlim([0 samples])
    yticks(min(win_percent2):(max(win_percent2)-min(win_percent2))/6:max(win_percent2))
    yline(2/3,'r')
    ylabel('Cumulative Win %')
    xlabel('Sample #')
    title('Weak Law of Large Numbers')
    drawnow limitrate;

    nexttile(2)
    histogram(win_percent(1:i),20)
    xline(2/3,'r')
    xlim([min(win_percent) max(win_percent)])
    ylim([0 samples/5])
    xticks(min(win_percent):(max(win_percent)-min(win_percent))/6:max(win_percent))
    ylabel('Count of samples')
    xlabel('Sample Win %')
    title('Central Limit Theorem')
    drawnow limitrate;
end

