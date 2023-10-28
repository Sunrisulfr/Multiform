function data_MFCMAES = MFCMAES(Tasks,ini_pop,gen,cp,reps,index,rem)
    %
    % Copyright (c) 2015, Yarpiz (www.yarpiz.com)
    % All rights reserved. Please read the "license.txt" for license terms.
    %
    % Project Code: YPEA108
    % Project Title: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    % Publisher: Yarpiz (www.yarpiz.com)
    % 
    % Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
    % 
    % Contact Info: sm.kalami@gmail.com, info@yarpiz.com
    %

    tic

    %% Problem Settings
    
    no_of_tasks=length(Tasks);
    D=zeros(1,no_of_tasks);
    
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
    
    
    fnceval_calls = zeros(1,reps);  
    calls_per_individual=zeros(1,pop);
    EvBestFitness = zeros(no_of_tasks*reps,gen);    % best fitness found
    TotalEvaluations=zeros(reps,gen);               % total number of task evaluations so fer
    bestobj=Inf(1,no_of_tasks);
    bestFncErrorValue = zeros(100,60);
    bestFitnessCrossTask = zeros(no_of_tasks,gen);
    
    load('groupInfo.mat');
    A = groupA(index);
    bounds = groupB(index);

    for i=1:length(Tasks)
%         Tasks(i).A = cell2mat(A{1}(i));
        Tasks(i).A = normrnd(0,1,Tasks(i).D_high,Tasks(i).D_func);
        Tasks(i).B_eff = cell2mat(bounds{1}(i));
    end
    
    
    
    for rep = 1:reps        
        disp(rep);
        
        nVar=D_multitask;                % Number of Unknown (Decision) Variables

        VarSize=[1 nVar];       % Decision Variables Matrix Size

        VarMin=-1;             % Lower Bound of Decision Variables
        VarMax= 1;             % Upper Bound of Decision Variables

        %% CMA-ES Settings
        % Maximum Number of Iterations
        MaxIt=gen;

        % Population Size (and Number of Offsprings)
        lambda=pop/no_of_tasks;

        % Number of Parents
        mu=round(lambda/2);

        % Parent Weights
        w=log(mu+0.5)-log(1:mu);
        w=w/sum(w);

        % Number of Effective Solutions
        mu_eff=1/sum(w.^2);

        % Step Size Control Parameters (c_sigma and d_sigma);
        sigma0=0.3*(VarMax-VarMin);
        cs=(mu_eff+2)/(nVar+mu_eff+5);
        ds=1+cs+2*max(sqrt((mu_eff-1)/(nVar+1))-1,0);
        ENN=sqrt(nVar)*(1-1/(4*nVar)+1/(21*nVar^2));

        % Covariance Update Parameters
        cc=(4+mu_eff/nVar)/(4+nVar+2*mu_eff/nVar);
        c1=2/((nVar+1.3)^2+mu_eff);
        alpha_mu=2;
        cmu=min(1-c1,alpha_mu*(mu_eff-2+1/mu_eff)/((nVar+2)^2+alpha_mu*mu_eff/2));
        hth=(1.4+2/(nVar+1))*ENN;

        %% Initialization

%         ps=cell(MaxIt,1);
%         pc=cell(MaxIt,1);
%         C=cell(MaxIt,1);
%         sigma=cell(MaxIt,1);
% 
%         ps{1}=zeros(VarSize);
%         pc{1}=zeros(VarSize);
%         C{1}=eye(nVar);
%         sigma{1}=sigma0;
% 
%         empty_individual.Position=[];
%         empty_individual.Step=[];
%         empty_individual.Cost=[];
% 
%         M=repmat(empty_individual,MaxIt,1);
%         M(1).Position=unifrnd(VarMin,VarMax,VarSize);
%         M(1).Step=zeros(VarSize);
%         M(1).Cost=fnceval(Tasks(1),M(1).Position,0,0);
%         
%         BestSol=M(1);
% 
%         BestCost=zeros(MaxIt,1);
        
        for nt = 1:no_of_tasks
            cmaes(nt) = CMAESsolver();
%             cmaes(nt).initialize(MaxIt,nVar);
            cmaes(nt).empty_individual.Position=[];
            cmaes(nt).empty_individual.Step=[];
            cmaes(nt).empty_individual.Cost=[];

            cmaes(nt).ps{1}=zeros([1 nVar]);
            cmaes(nt).pc{1}=zeros([1 nVar]);
            cmaes(nt).C{1}=eye(nVar);
            
            cmaes(nt).sigma{1}=sigma0;

            cmaes(nt).M = repmat(cmaes(nt).empty_individual,MaxIt,1);
            cmaes(nt).pop=repmat(cmaes(nt).empty_individual,lambda,1);

            cmaes(nt).M(1).Position=unifrnd(VarMin,VarMax,VarSize);
            cmaes(nt).M(1).Step=zeros(VarSize);
            cmaes(nt).M(1).Cost=cmaes(nt).evaluate(Tasks(nt),cmaes(nt).M(1).Position);
            cmaes(nt).BestSol=cmaes(nt).M(1);
        end
        
        
        inj_solution = [];
        %% CMA-ES Main Loop
        for g=1:MaxIt
            for nt = 1:no_of_tasks
                % Generate Samples
                for i=1:lambda
                    cmaes(nt).pop(i).Step=mvnrnd(zeros(VarSize),cmaes(nt).C{g});
                    cmaes(nt).pop(i).Position=cmaes(nt).M(g).Position+cmaes(nt).sigma{g}*cmaes(nt).pop(i).Step;
                    cmaes(nt).pop(i).Cost = cmaes(nt).evaluate(Tasks(nt),cmaes(nt).pop(i).Position);

                    % Update Best Solution Ever Found
                    if cmaes(nt).pop(i).Cost<cmaes(nt).BestSol.Cost
                        cmaes(nt).BestSol=cmaes(nt).pop(i);
                    end
                end
                
                % Sort Population
                Costs=[cmaes(nt).pop.Cost];
                [Costs, SortOrder]=sort(Costs);
                cmaes(nt).pop=cmaes(nt).pop(SortOrder);
                
%                 if ~isempty(inj_solution)
%                     V = 1:no_of_tasks;V(nt)=[];
%                     tmp_fvalue = inf;
%                     inj_label = -1;
%                     for j =1:length(V)
%                         test = repmat(cmaes(nt).empty_individual,1,1);
%                         test.Step = mvnrnd(zeros(VarSize),cmaes(nt).C{g});
%                         test.Position=inj_solution((nt-1)*(no_of_tasks-1)+j,:);
%                         test.Cost = cmaes(nt).evaluate(Tasks(nt),test.Position);
%                         if(test.Cost<tmp_fvalue)
%                             inj_label = V(j);
%                             tmp_fvalue = test.Cost;
%                             cmaes(nt).pop(lambda) = test;
%                         end
%                     end
%                 end

                % Save Results
                BestCost(g)=cmaes(nt).BestSol.Cost;

%                 % Display Results
%                 disp(['Iteration ' num2str(g) ': Best Cost = ' num2str(BestCost(g))]);
                
                bestobj(nt)=BestCost(g);
                
                EvBestFitness(nt+no_of_tasks*(rep-1),g)=bestobj(nt);

%                 % Exit At Last Iteration
%                 if g==MaxIt
%                     break;
%                 end

                % Update Mean
                cmaes(nt).M(g+1).Step=0;
                for j=1:mu
                    cmaes(nt).M(g+1).Step=cmaes(nt).M(g+1).Step+w(j)*cmaes(nt).pop(j).Step;
                end
                cmaes(nt).M(g+1).Position=cmaes(nt).M(g).Position+cmaes(nt).sigma{g}*cmaes(nt).M(g+1).Step;
                cmaes(nt).M(g+1).Cost=cmaes(nt).evaluate(Tasks(nt),cmaes(nt).M(g+1).Position);

                if cmaes(nt).M(g+1).Cost<cmaes(nt).BestSol.Cost
                    cmaes(nt).BestSol=cmaes(nt).M(g+1);
                end

                % Update Step Size
                cmaes(nt).ps{g+1}=(1-cs)*cmaes(nt).ps{g}+sqrt(cs*(2-cs)*mu_eff)*cmaes(nt).M(g+1).Step/chol(cmaes(nt).C{g})';
                cmaes(nt).sigma{g+1}=cmaes(nt).sigma{g}*exp(cs/ds*(norm(cmaes(nt).ps{g+1})/ENN-1))^0.3;

                % Update Covariance Matrix
                if norm(cmaes(nt).ps{g+1})/sqrt(1-(1-cs)^(2*(g+1)))<hth
                    hs=1;
                else
                    hs=0;
                end
                delta=(1-hs)*cc*(2-cc);
                cmaes(nt).pc{g+1}=(1-cc)*cmaes(nt).pc{g}+hs*sqrt(cc*(2-cc)*mu_eff)*cmaes(nt).M(g+1).Step;
                cmaes(nt).C{g+1}=(1-c1-cmu)*cmaes(nt).C{g}+c1*(cmaes(nt).pc{g+1}'*cmaes(nt).pc{g+1}+delta*cmaes(nt).C{g});
                for j=1:mu
                    cmaes(nt).C{g+1}=cmaes(nt).C{g+1}+cmu*w(j)*cmaes(nt).pop(j).Step'*cmaes(nt).pop(j).Step;
                end

                % If Covariance Matrix is not Positive Defenite or Near Singular
                [V, E]=eig(cmaes(nt).C{g+1});
                if any(diag(E)<0)
                    E=max(E,0);
                    cmaes(nt).C{g+1}=V*E/V;
                end

                for i=1:lambda
                    population(i) = Chromosome();
                    population(i).rnvec = cmaes(nt).pop(i).Position;
                    population(i).factorial_costs = inf(1,no_of_tasks);
                    population(i).factorial_costs(nt) = cmaes(nt).pop(i).Cost;
                    population(i).skill_factor = nt;
                end
                if g==1
                    Tasks(nt).Pop = population;
                elseif g~=1
                    Tasks(nt).Pop = [Tasks(nt).Pop population];
                end
            end
                        
            
            if(cp<0 && mod(g,10)==0)
                W = mDA_mapping(Tasks);
                for nt = 1:no_of_tasks
                    Tasks(nt).Pop(1:length(Tasks(nt).Pop)-1) = [];
                end
            end
            
            if(cp<0 && g>10)
                inj_solution=[];
                for nt = 1:no_of_tasks
                    s = [];
                    V = 1:no_of_tasks;V(nt)=[];
                    
                    for j =1:length(V)
                        his_bestSolution = [];
                        p2_tmp = inf;
                        for i = 1:lambda
                            if p2_tmp >cmaes(V(j)).pop(i).Cost
                                p2_tmp = cmaes(V(j)).pop(i).Cost;
                                his_bestSolution = cmaes(V(j)).pop(i).Position;
                            end
                        end
                        his_bestSolution = his_bestSolution(1:Tasks(V(j)).D_func);
                        
                        curr_len = Tasks(nt).D_func;
                        tmp_len = Tasks(V(j)).D_func;
                        
                        w_map=cell2mat(W(j+(nt-1)*length(V)));
                        if curr_len <= tmp_len
                            tmp_solution = (w_map*his_bestSolution')';
                            tmp_rnvec = tmp_solution(:,1:curr_len);
                        elseif  curr_len > tmp_len
                            his_bestSolution(:,tmp_len+1:curr_len) = 0;
                            tmp_rnvec = (w_map*his_bestSolution')';
                        end
                        tmp_rnvec = [];
                        tmp_rnvec(tmp_rnvec>1) =1;
                        tmp_rnvec(tmp_rnvec<-1)=-1;
                        if length(tmp_rnvec) < D_multitask
                            tmp_rnvec(length(tmp_rnvec)+1:D_multitask) =0;
                        end
                        solution = [];
                    end
                    inj_solution = [inj_solution; solution];
                end
            end
            
            disp(['MFEA Generation = ', num2str(g), ' best factorial costs = ', num2str(bestobj)]);
        end
        bestFitnessCrossTask = bestFitnessCrossTask + (EvBestFitness(1+no_of_tasks*(rep-1):rep*no_of_tasks, :));
    end
    
    data_MFCMAES.bestFitnessCrossTask = bestFitnessCrossTask;
    data_MFCMAES.wall_clock_time=toc;
    data_MFCMAES.EvBestFitness=EvBestFitness;
    data_MFCMAES.bestInd_data=[];
    data_MFCMAES.TotalEvaluations=[];
    
    %% Display Results

%     figure;
%     % plot(BestCost, 'LineWidth', 2);
%     semilogy(BestCost, 'LineWidth', 2);
%     xlabel('Iteration');
%     ylabel('Best Cost');
%     grid on;

end
