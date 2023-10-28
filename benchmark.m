function [Tasks, g1, g2] = benchmark(index,N_tasks, d_fun)
%BENCHMARK function
%   Input
%   - index: the index number of problem set
%
%   Output:
%   - Tasks: benchmark problem set
%   - g1: global optima of Task 1
%   - g2: global optima of Task 2

%     load('Tasks\Ackley.mat');
    load('MMM50.mat');
%     GO_Task = (floor((2*rand(1,50)-1)*100)/100);
%     GO_Task = zeros(1,5000);
% % 
%     Rotation_M = eye(5000);
%     Rotation_M=orth(randn(50,50));

%     fid=fopen('Rotation_M.txt','wt');%写入文件路径
%     [m,n]=size(Rotation_M);
%      for i=1:1:m
%         for j=1:1:n
%            if j==n
%              fprintf(fid,'%g\n',Rotation_M(i,j));
%           else
%             fprintf(fid,'%g\t',Rotation_M(i,j));
%            end
%         end
%     end
%     fclose(fid);

    D_eff = 50;
    D_func = d_fun;
    D_high = 50;
%     B_ori = ones(D_func, 2);            
%     B_ori(:,1) = -B_ori(:,1);
    B_eff = ones(D_eff,2);
    B_eff(:,1) = -B_eff(:,1);
    switch(index)
       
        case 1 % complete intersection with medium similarity, Ackley and Rastrigin
             
            for i = 1:N_tasks
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func(i);
                Tasks(i).D_high = D_high;
%                 Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = GO_Task;
                Tasks(i).Rotation_M = Rotation_M;
                Tasks(i).fnc = @(x)Sphere(x,Rotation_M,GO_Task,Tasks(i).D_eff);
                Tasks(i).A = zeros(D_high, D_eff);
                Tasks(i).Pop = 0;
            end 
        case 2 % complete intersection with medium similarity, Ackley and Rastrigin
             
            for i = 1:N_tasks
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func(i);
                Tasks(i).D_high = D_high;
%                 Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = GO_Task;
                Tasks(i).Rotation_M = Rotation_M;
                Tasks(i).fnc = @(x)Ackley(x,Rotation_M,GO_Task,Tasks(i).D_eff);
                Tasks(i).A = zeros(D_high, D_eff);
            end
             
 
        case 3 % complete intersection with low similarity, Ackley and Schwefel

            for i = 1:N_tasks
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func(i);
                Tasks(i).D_high = D_high;
%                 Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = GO_Task;
                Tasks(i).Rotation_M = Rotation_M;
                Tasks(i).fnc = @(x)Rastrigin(x,Rotation_M,GO_Task,Tasks(i).D_eff);
                Tasks(i).A = zeros(D_high, D_eff);
            end
 
        case 4 % partially intersection with high similarity, Rastrigin and Sphere
             
            for i = 1:N_tasks
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func(i);
                Tasks(i).D_high = D_high;
%                 Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = GO_Task;
                Tasks(i).Rotation_M = Rotation_M;
                Tasks(i).fnc = @(x)Weierstrass(x,Rotation_M,GO_Task,Tasks(i).D_eff);
                Tasks(i).A = zeros(D_high, D_eff);
            end
             
        case 5 % partially intersection with medium similarity, Ackley and Rosenbrock
             
            for i = 1:N_tasks
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func(i);
                Tasks(i).D_high = D_high;
%                 Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = GO_Task;
                Tasks(i).Rotation_M = Rotation_M;
                Tasks(i).fnc = @(x)Rosenbrock(x,Rotation_M,GO_Task,Tasks(i).D_eff);
                Tasks(i).A = zeros(D_high, D_eff);
            end
             
        case 6 % partially intersection with low similarity, Ackley and Weierstrass
             
            for i = 1:N_tasks
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func(i);
                Tasks(i).D_high = D_high;
%                 Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = GO_Task;
                Tasks(i).Rotation_M = Rotation_M;
                Tasks(i).fnc = @(x)Griewank(x,Rotation_M,GO_Task,Tasks(i).D_eff);
                Tasks(i).A = zeros(D_high, D_eff);
            end 
            
        case 7 % partially intersection with low similarity, Ackley and Weierstrass
             
            for i = 1:N_tasks
%                 Tasks(i).D_eff = D_eff;
%                 Tasks(i).D_func = D_func(i);
%                 Tasks(i).D_high = D_high;
% %                 Tasks(i).B_ori = B_ori;
%                 Tasks(i).B_eff = B_eff;
%                 Tasks(i).GO_Task = GO_Task;
%                 Tasks(i).Rotation_M = Rotation_M;
%                 Tasks(i).fnc = @(x)PermFunction(x,Rotation_M,GO_Task,Tasks(i).D_eff);
%                 Tasks(i).A = zeros(D_high, D_eff);
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func(i);
                Tasks(i).D_high = D_high;
%                 Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = GO_Task;
                Tasks(i).Rotation_M = Rotation_M;
                Tasks(i).fnc = @(x)Sphere(x,Rotation_M,GO_Task,Tasks(i).D_eff);
                Tasks(i).A = zeros(D_high, D_eff);
                Tasks(i).Pop = 0;
            end     
            
            
            
        case 8
            class = 10;
            D_eff = 15;
            D_func = 15;
            D_high = 0.5*class*(class-1);
            
            B_ori = ones(D_func, 2); 
            B_ori(:,1) = -1;
            
            B_eff = zeros(D_eff,2);
            B_eff(:,2) = 1;
            
            for i = 1:N_tasks
                Tasks(i).class = class;
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func;
                Tasks(i).D_high = D_high;
                Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = 1*ones(1,D_func);
                Tasks(i).fnc = @(x)SVM(x);
                Tasks(i).A = zeros(D_high, D_eff);
            end
            
        case 18 % no intersection with medium similarity, Griewank and Weierstrass
             
             
            load('Tasks\NI_M.mat')
            dim = 50;
            Tasks(1).dims = dim;
            Tasks(1).fnc = @(x)Griewank(x,Rotation_Task1,GO_Task1);
            Tasks(1).Lb=-100*ones(1,dim);
            Tasks(1).Ub=100*ones(1,dim);
             
            Tasks(2).dims = dim;
            Tasks(2).fnc = @(x)Weierstrass(x,Rotation_Task2,GO_Task2);
            Tasks(2).Lb=-0.5*ones(1,dim);
            Tasks(2).Ub=0.5*ones(1,dim);
             
            g1 = GO_Task1;
            g2 = GO_Task2;
        case 9 % no overlap with low similarity, Rastrigin and Schwefel
             for i = 1:N_tasks
                Tasks(i).D_eff = D_eff;
                Tasks(i).D_func = D_func(i);
                Tasks(i).D_high = D_high;
%                 Tasks(i).B_ori = B_ori;
                Tasks(i).B_eff = B_eff;
                Tasks(i).GO_Task = GO_Task;
                Tasks(i).Rotation_M = Rotation_M;
                Tasks(i).fnc = @(x)Sphere(x,Rotation_M,GO_Task,Tasks(i).D_eff);
                Tasks(i).A = zeros(D_high, D_eff);
                Tasks(i).Pop = 0;
            end 
            
     case 0 % complete intersection with high similarity, Griewank and Rastrigin
%             load('Tasks\CI_H.mat')  % loading data from folder .\Tasks
             
            D_eff = 50;
            D_func = 50;
            D_high = 5000;
            B_ori = ones(D_func, 2);            
            B_ori(:,1) = -B_ori(:,1);
            B_eff = ones(D_eff,2);
             
            rand_M = normrnd(0, 1, D_high, D_high);
            [Rotation_M, R] = qr(rand_M);
            
            orth
            GO_Task = (floor((1*rand(1,5000)-0.5)*100)/100);

            save ('Tasks\Ackley.mat', 'Rotation_M','GO_Task');
             
             
            rand_M = normrnd(0, 1, 1000, 1000);  
            [Rotation_M, R] = qr(rand_M); 
             
            Tasks(1).D_eff = D_eff;
            Tasks(1).D_func = D_func;
            Tasks(1).D_high = D_high;
            Tasks(1).B_ori = B_ori;
            Tasks(1).B_eff = B_eff;
            Tasks(1).GO_Task = GO_Task;
            Tasks(1).fnc = @(x)Sphere(x,GO_Task,B_ori,Rotation_M);
            Tasks(1).A = zeros(D_high, D_eff);  
    end
    
%     Tasks(i).D_eff = D_eff;
%     Tasks(5).D_func = 5000;
%     Tasks(5).D_high = 5000;
% %                 Tasks(i).B_ori = B_ori;
%     Tasks(i).B_eff = B_eff;
%     Tasks(i).GO_Task = GO_Task;
%     Tasks(i).Rotation_M = Rotation_M;
%     Tasks(i).fnc = @(x)Sphere(x,Rotation_M,GO_Task,Tasks(i).D_eff);
%     Tasks(i).A = zeros(D_high, D_eff);
%     Tasks(i).Pop = 0;
                
end