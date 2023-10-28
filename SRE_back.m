function data_SRE = SRE(Task,pop,gen,selection_process,p_il,reps,rem,groupA,groupBounds,EA,N_RM)
%SOEA function: implementation of SOEA algorithm
    tic         
    if mod(pop,2) ~= 0
        pop = pop + 1;
    end   
    if rem == 0
        D = Task.D_high;
    else
        D = Task.D_func;
    end
    
    options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',2);  % settings for individual learning
    
    
    fnceval_calls = zeros(1,reps); 
    calls_per_individual=zeros(1,pop);
    EvBestFitness = zeros(reps,gen);    % best fitness found
    TotalEvaluations=zeros(reps,gen);   % total number of task evaluations so fer
    bestFitnessCrossTask = [];
        
    for rep = 1:reps   
        disp(rep);

        fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
        TotalEvaluations(rep,1)=fnceval_calls(rep);    
        bestobj=0;
        EvBestFitness(rep,1) = bestobj;  
        
        M = cell2mat(groupA(rep));
        bounds = cell2mat(groupBounds(rep));
    
%         VarSize=[1 D];   % Decision Variables Matrix Size
%         beta_min=0.2;   % Lower Bound of Scaling Factor
%         beta_max=0.8;   % Upper Bound of Scaling Factor

        generation =1;
        for itr_sre = 1: N_RM
            
            Task.A = M(:, 1+(itr_sre-1)*Task.D_func : itr_sre*Task.D_func);
            Task.B_eff = bounds;
            for i = 1 : pop
                population(i) = Chromosome();
                population(i) = initialize(population(i),D);
            end
            for i = 1 : pop
                [population(i),calls_per_individual(i)] = evaluate_SOO(population(i),Task,p_il,options);
            end
            if itr_sre~=1
                population(1) = bestInd_data(rep);
            end
            bestobj = min([population.factorial_costs]);
            disp(['SOO Generation ', num2str(generation), ' best objective = ', num2str(bestobj)])

            while generation < itr_sre*(gen/N_RM)      
                if EA ==1 %% DE
                    lb=-ones(1,D);
                    ub=ones(1,D);
                    pCR=0.9;
                    F=0.5;
                    generation = generation + 1;
                    count=1;
                    for i = 1 : pop
                        x=population(i).rnvec;
                        A = randperm(pop);

                        A(A==i)=[];
                        p1=A(1);
                        p2=A(2);
                        p3=A(3);

                        y=population(p1).rnvec+F*(population(p2).rnvec-population(p3).rnvec);

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

                        count=count+1;
                    end  
                elseif EA==2
                    mu = 10; % Index of Simulated Binary Crossover (tunable)
                    sigma = 0.02; % standard deviation of Gaussian Mutation model (tunable)
                    generation = generation + 1;
                    indorder = randperm(pop);
                    count=1;
                    for i = 1 : pop/2     
                        p1 = indorder(i);
                        p2 = indorder(i+(pop/2));
                        child(count)=Chromosome();
                        child(count+1)=Chromosome();
                        u = rand(1,D);
                        cf = zeros(1,D);
                        cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                        cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                        child(count) = crossover(child(count),population(p1),population(p2),cf);
                        child(count+1) = crossover(child(count+1),population(p2),population(p1),cf);
                        if rand(1) < 0.1
                            child(count)=mutate(child(count),child(count),D,sigma);
                            child(count+1)=mutate(child(count+1),child(count+1),D,sigma);
                        end            
                        count=count+2;
                    end     
                end
                for i = 1 : pop            
                    [child(i),calls_per_individual(i)] = evaluate_SOO(child(i),Task,p_il,options);           
                end      

                fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
                TotalEvaluations(rep,generation)=fnceval_calls(rep);

                intpopulation(1:pop)=population;
                intpopulation(pop+1:2*pop)=child;
                [xxx,y]=sort([intpopulation.factorial_costs]);
                intpopulation=intpopulation(y);
                for i = 1:2*pop
                    intpopulation(i).scalar_fitness=1/i;
                end
                if intpopulation(1).factorial_costs<=bestobj
                    bestobj=intpopulation(1).factorial_costs;
                    bestInd_data(rep)=intpopulation(1);
                end
                EvBestFitness(rep,generation)=bestobj;

                if strcmp(selection_process,'elitist')
                    [xxx,y]=sort(-[intpopulation.scalar_fitness]);
                    intpopulation=intpopulation(y);
                    population=intpopulation(1:pop);            
                elseif strcmp(selection_process,'roulette wheel')
                    for i=1:pop
                        population(i)=intpopulation(RouletteWheelSelection([intpopulation.scalar_fitness]));
                    end    
                end
                disp(['SOO Generation ', num2str(generation), ' best objective = ', num2str(bestobj)])
            end
            
            [bestobj, bestpop] = min([population.factorial_costs]);
            bestInd_data(rep) = population(bestpop);
        end
         
    end
    data_SOO.wall_clock_time=toc;
    data_SOO.EvBestFitness=EvBestFitness;
    data_SOO.bestInd_data=bestInd_data;
    data_SOO.TotalEvaluations=TotalEvaluations;
end