classdef CMAESsolver    
    properties
        ps;
        pc;
        C;
        sigma;
        empty_individual;
        M;
        BestSol;
        pop;
    end    
    methods
        function object = initialize(object,MaxIt,nVar)
%             object.empty_individual.Position=[];
%             object.empty_individual.Step=[];
%             object.empty_individual.Cost=[];
%             
%             object.ps{1}=zeros([1 nVar]);
%             object.pc{1}=zeros([1 nVar]);
%             object.C{1}=eye(nVar);
%             
% %             object.sigma{1}=sigma0;
%             
%             object.M = repmat(object.empty_individual,MaxIt,1);
%             object.pop=repmat(object.empty_individual,lambda,1);
        end
        
        function cost = evaluate(object,Task,position) 
            [cost,xxx,xxx]=fnceval(Task,position,0,0);
        end
    end
end