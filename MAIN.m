% This MATLAB R2014b code is for EVOLUTIONARY MULTITASKING across minimization problems. 
% For maximization problems, multiply objective function by -1.
 
% Settings of simulated binary crossover (SBX) in this code is Pc = 1, 
% and probability of variable sawpping = 0.  
%% Calling the solvers
% For large population sizes, consider using the Parallel Computing Toolbox
% of MATLAB.
% Else, program can be slow. 

clc;
pop_M=100; % population size 100
pop_S = pop_M;
gen=500; % generation count 1000
selection_pressure = 'elitist'; % choose either 'elitist' or 'roulette wheel'
p_il = 0; % probability of individual learning (BFGA quasi-Newton Algorithm) --> Indiviudal Learning is an IMPORTANT component of the MFEA.
rmp=0; % random mating probability
reps=20; % repetitions 20

dataMean = [];
dataMeanFull = [];
dataStd = [];

N_fun = 6;
N_tasks =5;
D_func = [50 50 50 50 50]; % that is d

% scale = 0.5;
% groupA = cell(1,N_fun);
% groupB = cell(1,N_fun);
% for index = 1:N_fun
%     
%     Tasks = benchmark(index,N_tasks,D_func);
%     A = cell(1,N_tasks);
%     B = cell(1,N_tasks);
%     for i=1:length(Tasks)
%         Tasks(i).A = normrnd(0,1,Tasks(i).D_high,Tasks(i).D_func);
%         A(i) = {Tasks(i).A};
%         
%         bounds = scale*ones(Tasks(i).D_func, 2); 
%         bounds(:, 1) = -bounds(:, 1);
%         Tasks(i).B_eff = bounds;
%         B(i) = {Tasks(i).B_eff};
%     end
%     
%     groupA(index) = {A};
%     groupB(index) = {B};
% end
% save('groupInfo.mat','groupA','groupB');

for EA =1:1    %1=DE 2=GA
    for index = 2:7
%     for it_func = 1:length(D_func)
        Tasks = benchmark(index,N_tasks,D_func);

%         Tasks = benchmark(index,N_tasks,D_func(it_func)*ones(1,length(D_func)));
%         for i=1:length(Tasks)
%             Tasks(i).A = normrnd(0,1,Tasks(i).D_high,Tasks(i).D_func);
%             bounds = scale*ones(Tasks(i).D_func, 2); 
%             bounds(:, 1) = -bounds(:, 1);
%             Tasks(i).B_eff = bounds;
%         end

        for i = 1:1
            if i ==1
                rem=0;
%                     data_SOO(index)= PSO(Tasks(1),pop_M,gen,p_il,reps);
                data_SOO(index)=SOEA(Tasks(1),pop_M,gen,selection_pressure,p_il,reps,EA); 
                dataMean = [dataMean;mean(data_SOO(index).EvBestFitness)];
                dataStd = [dataStd;std(data_SOO(index).EvBestFitness)];
            elseif i==2
                rmp=0;rem=1;
%                     data_MFDE(index)=MFPSO(Tasks,pop_M,gen,rmp,p_il,reps,index,rem,EA);
                data_MFDE(index)=MFEA(Tasks(1),pop_M,gen,selection_pressure,rmp,p_il,reps,index,rem,EA);  
                dataMean = [dataMean;mean(data_MFDE(index).EvBestFitness)];
                dataStd = [dataStd;std(data_MFDE(index).EvBestFitness)];
            elseif i ==3
                rmp=0.3;rem=1;
%                     data_MFDE(index)=MFPSO(Tasks,pop_M,gen,rmp,p_il,reps,index,rem,EA);
                data_MFDE(index)=MFEA(Tasks,pop_M,gen,selection_pressure,rmp,p_il,reps,index,rem,EA);  
                dataMean = [dataMean;mean(data_MFDE(index).bestFitnessCrossTask)];
                dataStd = [dataStd;std(data_MFDE(index).bestFitnessCrossTask)];
            elseif i==4
                rmp=-1;rem=1;
%                     data_MFDE(index)=MFPSO(Tasks,pop_M,gen,rmp,p_il,reps,index,rem,EA);
%                     data_MFDE(index) = MFCMAES(Tasks,pop_M,gen,rmp,reps,index,rem)            
%                 data_MFDE(index)=MFEA(Tasks,pop_M,gen,selection_pressure,rmp,p_il,reps,index,rem,EA);  
                data_MFDE(index)=MFEA_mf(Tasks,pop_M,gen,selection_pressure,rmp,p_il,reps,index,rem,EA);  
                dataMeanFull = [dataMeanFull;data_MFDE(index).bestFitnessCrossTask];
                dataMean = [dataMean;mean(data_MFDE(index).bestFitnessCrossTask)];
                dataStd = [dataStd;std(data_MFDE(index).bestFitnessCrossTask)];
            elseif i==5
                rmp=-1;rem=1;
                data_MFDE(index)=SRE(Tasks(1),pop_M,gen,selection_pressure,rmp,p_il,reps,index,rem,EA,N_tasks);  
            end
        end
    end  
%     end
end


% save('result.mat','data_MFDE');
