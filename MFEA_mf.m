function data_MFDE = MFEA_mf(Tasks,ini_pop,gen,selection_process,cp,p_il,reps,index,rem,EA)
%MFEA function: implementation of MFEA algorithm
    tic 
    no_of_tasks=length(Tasks);
    D=zeros(1,no_of_tasks);
    Tasks(5).D_func = 50;
    pop = ini_pop;
    while mod(pop,no_of_tasks) ~= 0
        pop = pop + 1;
        ini_pop = pop;
    end
         
    for i=1:no_of_tasks
        if rem == 0
            D(i)=Tasks(i).D_high;
        else
            D(i)=Tasks(i).D_func;
        end
    end    
    D_multitask=max(D);

%     options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',2);  % settings for individual learning
     
    fnceval_calls = zeros(1,reps);  
    calls_per_individual=zeros(1,pop);
    EvBestFitness = zeros(no_of_tasks*reps,gen);    % best fitness found
    TotalEvaluations=zeros(reps,gen);               % total number of task evaluations so fer
    bestobj=Inf(1,no_of_tasks);
    bestFncErrorValue = zeros(100,60);
    bestFitnessCrossTask = [];
    
    
%     load('groupInfo.mat');
%     A = groupA(index);
%     bounds = groupB(index);
% 
%     for i=1:length(Tasks)
%         Tasks(i).A = cell2mat(A{1}(i));
%         Tasks(i).B_eff = cell2mat(bounds{1}(i));
%     end
    
    for rep = 1:reps        
        disp(rep);
        
        for i=1:length(Tasks)
            Tasks(i).A = normrnd(0,1,Tasks(i).D_high,Tasks(i).D_func);
            bounds = 0.5*ones(Tasks(i).D_func, 2); 
            bounds(:, 1) = -bounds(:, 1);
            Tasks(i).B_eff = bounds;
            for p=1:20
                pop_idx =(i-1)*ini_pop/length(Tasks)+p;
                population(pop_idx) = Chromosome();
                population(pop_idx) = initialize(population(pop_idx),D(i));
                population(pop_idx).skill_factor=0;
            end
        end
        
%         for i = 1 : pop
%             population(i) = Chromosome();
%             population(i) = initialize(population(i),D_multitask);
%             population(i).skill_factor=0;
%         end
        
        task_label = 1;
        for i=1:pop
            population(i).skill_factor = task_label;
            if mod(i,pop/no_of_tasks) ==0
                task_label= task_label +1;
            end
        end
        
        for i = 1 : pop
            [population(i),calls_per_individual(i)] = evaluate(population(i),Tasks,p_il,no_of_tasks);
            
        end
        
        for i = 1:no_of_tasks
            P= find([population.skill_factor]==i);
            Tasks(i).Pop = population(P);
        end

        fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
        TotalEvaluations(rep,1)=fnceval_calls(rep);
        
        
        factorial_cost=zeros(1,pop);
        for i = 1:no_of_tasks
            for j = 1:pop
                factorial_cost(j)=population(j).factorial_costs(i);
            end
            [xxx,y]=sort(factorial_cost);
            population=population(y);
            for j=1:pop
                population(j).factorial_ranks(i)=j; 
            end
            bestobj(i)=population(1).factorial_costs(i);
            EvBestFitness(i+no_of_tasks*(rep-1),1)=bestobj(i);
            bestInd_data(rep,i)=population(1);
        end
        bestobj
        bestobj_old = bestobj;
%       
        %resource_ratio
        resource_r =  ones(1,no_of_tasks)./no_of_tasks;
        omega_k = zeros(1,no_of_tasks);
        R_k =  zeros(1,no_of_tasks);
        re_R = zeros(1,no_of_tasks);
        sun_pop_runs = pop/2/no_of_tasks*ones(1,no_of_tasks);
        
        
        
        
        
        

%         for i=1:pop
%             population(i).skill_factor = unidrnd(no_of_tasks);
%             [xxx,yyy]=min(population(i).factorial_ranks);
%             x=find(population(i).factorial_ranks == xxx);
%             equivalent_skills=length(x);
%             if equivalent_skills>1
%                 population(i).skill_factor=x(1+round((equivalent_skills-1)*rand(1)));
%                 tmp=population(i).factorial_costs(population(i).skill_factor);
%                 population(i).factorial_costs(1:no_of_tasks)=inf;
%                 population(i).factorial_costs(population(i).skill_factor)=tmp;
%             else
%                 population(i).skill_factor=yyy;
%                 tmp=population(i).factorial_costs(population(i).skill_factor);
%                 population(i).factorial_costs(1:no_of_tasks)=inf;
%                 population(i).factorial_costs(population(i).skill_factor)=tmp;
%             end
%         end
%         for i = 1:no_of_tasks
%             Tasks(i).Pop = population;
%         end
        
        generation=1;
        while generation < gen
            
            
            
            
            pop = length(population);
            clear child;
            if EA ==1 %%Differetial Evolution 
                pCR=0.9;
                F=0.5;
                generation = generation + 1;
                count=1;

                class=[];
                group=cell(1,no_of_tasks);
                for idx_task = 1:no_of_tasks
                    for j = 1:pop
                        if population(j).skill_factor == idx_task
                            class = [class, j];
                        end
                    end
                    group(idx_task) = {class};
                    class=[];
                end

                asf=[];
                for i = 1:pop
                    x=population(i).rnvec;
                    asf = cell2mat(group(population(i).skill_factor));
                    asf = asf(randperm(length(asf)));

                    for j=1:length(asf)
                        if asf(j)==i
                            asf(j)=[];
                            break;
                        end
                    end
                    p1=asf(1);
                    p2=asf(2);
                    p3=asf(3);
                    
                    y=population(p1).rnvec+F*(population(p2).rnvec-population(p3).rnvec); 
                    lb=-ones(1,D(population(i).skill_factor));
                    ub=ones(1,D(population(i).skill_factor));
                    y=max(y,lb);
                    y=min(y,ub);

                    z=zeros(size(x)); 
                    j0=randi([1,numel(x)]);
                    for j=1:numel(x) 
                        if j==j0 || rand<=pCR 
                            z(j)=y(j);
                        else
                            z(j)=x(j);
                        end
                    end

                    child(count)=Chromosome();
                    child(count).rnvec=z;

                    child(count).skill_factor=population(i).skill_factor;

                    count=count+1;
                end
                
            elseif EA ==2 %%      Genetic Algorithm
                mu = 10; % Index of Simulated Binary Crossover (tunable)
                sigma = 0.02; % standard deviation of Gaussian Mutation model (tunable)                
                generation = generation + 1;
                
                count=1;
                
                
                
                for nt = 1:no_of_tasks
                    P = find([population.skill_factor]==nt);
                    indorder = randperm(length(P));
                    P = P(indorder);
                    
                    
                    
%                    Resource allocation with constant population size                 
%                     run_indorder = zeros(1,sun_pop_runs(nt)*2);
%                     for i = 1:ceil(sun_pop_runs(nt)*2/length(P))
%                         if i == ceil(sun_pop_runs(nt)*2/length(P)) && mod(sun_pop_runs(nt),length(P)/2) ~= 0
%                             indorder = randperm(length(P));
%                             
% %                             indorder(1:mod(sun_pop_runs,10))
% %                             run_indorder(i*10-9: sun_pop_runs)
%                             run_indorder(i*length(P)/2-(length(P)/2-1): sun_pop_runs(nt)) = indorder(1:mod(sun_pop_runs(nt),length(P)/2));
%                             
%                             run_indorder(i*length(P)/2-(length(P)/2-1)+sun_pop_runs(nt):sun_pop_runs(nt)*2) = indorder(length(P)/2+1:mod(sun_pop_runs(nt),length(P)/2)+length(P)/2);
%                         else
%                             indorder = randperm(length(P));
%                             run_indorder(i*length(P)/2-(length(P)/2-1): i*length(P)/2) = indorder(1:length(P)/2);
%                          run_indorder(i*length(P)/2-(length(P)/2-1)+sun_pop_runs(nt): i*length(P)/2+sun_pop_runs(nt)) = indorder(length(P)/2+1:length(P));
%                         end
%                         
%                     end                
                    
%                     for i = 1 : sun_pop_runs(nt)
%                         p1 = P(run_indorder(i));
%                         p2 = P(run_indorder(i+sun_pop_runs(nt)));




                    for i = 1 : length(P)/2  
                        p1 = P(i);
                        p2 = P(i+length(P)/2);  
                        child(count)=Chromosome();
                        child(count+1)=Chromosome();
                        corss_flag = rand(1);

                        if (population(p1).skill_factor == population(p2).skill_factor)

                            u = rand(1,D(nt));
                            cf = zeros(1,D(nt));
                            cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                            cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                            child(count) = crossover(child(count),population(p1),population(p2),cf);
                            child(count+1) = crossover(child(count+1),population(p2),population(p1),cf);
                            if rand(1) < 0.1 
                                child(count)=mutate(child(count),child(count),D(nt),sigma);
                                child(count+1)=mutate(child(count+1),child(count+1),D(nt),sigma);
                            end      

                            sf1=1+round(rand(1));
                            sf2=1+round(rand(1));

                            if sf1 == 1
                                child(count).skill_factor=population(p1).skill_factor;
                            else
                                child(count).skill_factor=population(p2).skill_factor;
                            end
                            if sf2 == 1
                                child(count+1).skill_factor=population(p1).skill_factor;
                            else
                                child(count+1).skill_factor=population(p2).skill_factor;
                            end

                            child(count).rnvec(child(count).rnvec>1)=1;
                            child(count).rnvec(child(count).rnvec<-1)=-1;
                            child(count+1).rnvec(child(count+1).rnvec>1)=1;
                            child(count+1).rnvec(child(count+1).rnvec<-1)=-1;

                            % variable swap
                            swap_indicator = (rand(1,D(nt)) >= 0.5);
                            temp = child(count+1).rnvec(swap_indicator);
                            child(count+1).rnvec(swap_indicator) = child(count).rnvec(swap_indicator);
                            child(count).rnvec(swap_indicator) = temp;

                        else
                            ccc = 11
                            child(count)=mutate(child(count),population(p1),D(nt),sigma);
                            child(count).skill_factor=population(p1).skill_factor;
                            child(count+1)=mutate(child(count+1),population(p2),D(nt),sigma);
                            child(count+1).skill_factor=population(p2).skill_factor;
                        end
                        count=count+2;
                    end 
                   
                end
            end
            
            for i = 1 : pop            
                [child(i),calls_per_individual(i)] = evaluate(child(i),Tasks,p_il,no_of_tasks);
                [factorial_v,factorial_indx] = min(child(i).factorial_costs);
                if(cp<0)
                    Tasks(factorial_indx).Pop = [Tasks(factorial_indx).Pop child(i)]; 
                end
            end             
            fnceval_calls(rep)= fnceval_calls(rep) + sum(calls_per_individual);
            TotalEvaluations(rep,generation)=fnceval_calls(rep);
              
 
            intpopulation(1:pop)=population;
            intpopulation(pop+1:2*pop)=child;
            
            factorial_cost=zeros(1,2*pop);
            for i = 1:no_of_tasks
                for j = 1:2*pop
                    
                    factorial_cost(j)=intpopulation(j).factorial_costs(i);
                end
                [xxx,y]=sort(factorial_cost);
                intpopulation=intpopulation(y);
                for j=1:2*pop
                    intpopulation(j).factorial_ranks(i)=j;
                end
                if intpopulation(1).factorial_costs(i)<=bestobj(i)
                    bestobj(i)=intpopulation(1).factorial_costs(i);
                    bestInd_data(rep,i)=intpopulation(1);
                end
                EvBestFitness(i+no_of_tasks*(rep-1),generation)=bestobj(i);
                 
                interval = pop;
                if mod(fnceval_calls(rep),interval)==0
                     bestFncErrorValue(fnceval_calls(rep)/interval,1)=fnceval_calls(rep);
                     bestFncErrorValue(fnceval_calls(rep)/interval,i+no_of_tasks*(rep-1)+1)=bestobj(i);
                end 
            end
            
            
%             % resource ratio update        

            if generation >25
                for i = 1:no_of_tasks
                    R_k(i)  = (bestobj_old(i) - bestobj(i))/(bestobj_old(i) +0.0001);
    %                 omega_k(i) = (bestobj_old(i) - bestobj(i))/(bestobj_old(i) +0.0001);
                end
                alpha = 2;
                for i = 1:no_of_tasks
%               omega_k(i) = omega_k(i) + alpha*(R_k(i) - mean(R_k))*(1-resource_r(i));
                omega_k(i) = omega_k(i) + alpha*R_k(i)-alpha*resource_r(i)*sum(R_k);
%               omega_k(i) = omega_k(i) + alpha*(R_k(i) - mean(R_k));
                end
                for i = 1:no_of_tasks
                    resource_r(i) = exp(omega_k(i))/sum(exp(omega_k));
                end
                resource_r
                bestobj_old = bestobj;
            end
            
            for nt = 1:no_of_tasks
                if nt == no_of_tasks
                    sun_pop_runs(nt) = pop/2 - sum(sun_pop_runs(1:no_of_tasks-1));
                    if sun_pop_runs(nt) == 1
                        [mm,pp] = max(sun_pop_runs);
                        sun_pop_runs(pp)=sun_pop_runs(pp) - 1;
                        sun_pop_runs(nt) = 2;
                    end
                else
                    sun_pop_runs(nt) = round(pop/2*resource_r(nt));
                    if sun_pop_runs(nt) == 1
                        sun_pop_runs(nt) = 2;
                    end
                end
            end
            sun_pop_runs
            
            
            
            
%              Old selection algorithm            
%             for i=1:2*pop
%                 [xxx,yyy]=min(intpopulation(i).factorial_ranks);
%                 intpopulation(i).skill_factor=yyy;
%                 intpopulation(i).scalar_fitness=1/xxx;
%             end   

            if strcmp(selection_process,'elitist')
                
%               New selection algorithm 
                t=1;
                for nt = 1:no_of_tasks
                    A = find([intpopulation.skill_factor]==nt);
                    population_A = intpopulation(A);
                    lenA = length(A);
                    
                    for i = 1:lenA
                         population_A(i).scalar_fitness=population_A(i).factorial_ranks(nt);
                    end
                    [xxx,y]=sort([population_A.scalar_fitness]);
                    population_A = population_A(y);
                    for i = 1:sun_pop_runs(nt)*2
                        if i > lenA
                            population(t+i-1) = Chromosome();
                            population(t+i-1) = initialize(population(t+i-1),D(nt));
                            population(t+i-1).skill_factor=nt;
                            [ population(t+i-1),calls_per_individual(i)] = evaluate( population(t+i-1),Tasks,p_il,no_of_tasks);
                        else
                            population(t+i-1) = population_A(i);
                        end
                    end
                    t = t+sun_pop_runs(nt)*2;
                    clear population_A
                end
                
                
%               Old selection algorithm                 
%                 [xxx,y]=sort(-[intpopulation.scalar_fitness]);
%                 intpopulation=intpopulation(y);
%                 population=intpopulation(1:ini_pop); 

            elseif strcmp(selection_process,'roulette wheel')
                for i=1:no_of_tasks
                    skill_group(i).individuals=intpopulation([intpopulation.skill_factor]==i);
                end
                count=0;
                while count<pop
                    count=count+1;
                    skill=mod(count,no_of_tasks)+1;
                    population(count)=skill_group(skill).individuals(RouletteWheelSelection([skill_group(skill).individuals.scalar_fitness]));
                end     
            end            
            
            if(cp<0 && mod(generation,1)==0&& generation<25)
                W = mDA_mapping(Tasks);
%                 for nt = 1:no_of_tasks
%                     Tasks(nt).Pop(1:length(Tasks(nt).Pop)-1) = [];
%                 end
            end
            
            p_best = inf;
            
            if(cp<0 && generation>=1&& generation<25)
                inj_solution=[];
                for nt = 1:no_of_tasks
                    solution = [];
                    V = 1:no_of_tasks;V(nt)=[];
                    
                    for j =1:length(V)
                        curr_pop=[];
                        his_pop=[];
                        his_bestSolution = [];
                        p2_tmp = inf;
                        for i = 1:pop
                            if population(i).skill_factor == nt
                                curr_pop = [curr_pop; population(i).rnvec];
                            elseif population(i).skill_factor == V(j)
                                his_pop = [his_pop; population(i).rnvec];
                                if p2_tmp > population(i).factorial_costs(1,V(j))
                                    p2_tmp = population(i).factorial_costs(1,V(j));
                                    his_bestSolution = population(i).rnvec;
                                end
                                if p_best > population(i).factorial_costs(1,V(j))
                                    p_best = population(i).factorial_costs(1,V(j));
                                end
                            end 
                        end
                        curr_pop = curr_pop(:,1:Tasks(nt).D_func);
                        his_pop = his_pop(:,1:Tasks(V(j)).D_func);
                        his_bestSolution = his_bestSolution(1:Tasks(V(j)).D_func);
                        curr_len = size(curr_pop, 2);
                        tmp_len = size(his_pop, 2);
                        
                        w=cell2mat(W(j+(nt-1)*length(V)));
                        if curr_len <= tmp_len
                            tmp_solution = (w*his_bestSolution')';
                            tmp_rnvec = tmp_solution(:,1:curr_len);
                        elseif  curr_len > tmp_len
                            his_bestSolution(:,tmp_len+1:curr_len) = 0;
                            tmp_rnvec = (w*his_bestSolution')';
                        end
                        tmp_rnvec(tmp_rnvec>1) =1;
                        tmp_rnvec(tmp_rnvec<-1)=-1;
                        if length(tmp_rnvec) < D_multitask
                            tmp_rnvec(length(tmp_rnvec)+1:D_multitask) =0;
                        end
                        solution = [solution; tmp_rnvec];
                    end
                    inj_solution = [inj_solution; solution];
                end

                for nt = 1:no_of_tasks
                    A = find([population.skill_factor]==nt);
                    V = 1:no_of_tasks;V(nt)=[];
                    tmp_fvalue = inf;
                    tmp_best = population(A(1)).factorial_costs(1,nt);
                    
                    inj_label = -1;
                    for j =1:length(V)
                        test=Chromosome();
                        test.rnvec=inj_solution((nt-1)*(no_of_tasks-1)+j,:);
                        if nt~=5
                            test.rnvec = test.rnvec(1,1:D(nt));
                        end
                        
                        test.skill_factor = nt;
                        [test,b] = evaluate(test,Tasks,p_il,no_of_tasks);
                        if(test.factorial_costs(1,nt)<tmp_fvalue)
                            inj_label = V(j);
                            tmp_fvalue = test.factorial_costs(1,nt);
                            population(A(length(A))) = test;
                        end
                        if test.factorial_costs(1,nt)< tmp_best
                            tmp_best = test.factorial_costs(1,nt);
%                             if tmp_best < bestobj_old(nt)
%                                 bestobj_old(nt) = tmp_best;
%                             end
                            disp(['Wow, from ',num2str(V(j)),' to ',num2str(nt)]);
                        end
                    end
%                         population(A(length(A)-j+1)).rnvec = inj_solution((nt-1)*(no_of_tasks-1)+j,:);
%                         [population(A(length(A)-j+1)),xx] = evaluate(population(A(length(A)-j+1)),Tasks,p_il,no_of_tasks,options);  
                end
            end

            disp(['MFEA Generation = ', num2str(generation), ' best factorial costs = ', num2str(bestobj)]);
        end 
        
        
        
        
        
        bestFitnessCrossTask = [bestFitnessCrossTask; min(EvBestFitness(1+no_of_tasks*(rep-1):rep*no_of_tasks, :))];
    end
  
    dlmwrite(['MTSOO_P',num2str(index),'.txt'],bestFncErrorValue,'precision',6);
    data_MFDE.bestFitnessCrossTask = bestFitnessCrossTask;
    data_MFDE.wall_clock_time=toc;
    data_MFDE.EvBestFitness=EvBestFitness;
    data_MFDE.bestInd_data=bestInd_data;
    data_MFDE.TotalEvaluations=TotalEvaluations;
     
%     for i=1:no_of_tasks
%         figure(i)
%         hold on
%         plot(EvBestFitness(i,:));
%         xlabel('GENERATIONS 2Task')
%         ylabel(['TASK ', num2str(i), ' OBJECTIVE'])
%         legend('MFEA')
%     end
end