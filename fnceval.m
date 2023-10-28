function [objective,rnvec,funcCount] = fnceval(Task,rnvec,p_il)
%FNCEVAL function: evaluate function in the unified search space
% decoding
    if size(rnvec,2) ~= Task.D_high %random embedding
        var = rnvec(1:Task.D_func);
        minrange=Task.B_eff(:,1)';
        maxrange=Task.B_eff(:,2)';
        y = maxrange - minrange;
        vars = maxrange.*var;
        nvars = vars(1:Task.D_func);
        vars = (Task.A*nvars')';
 
        minrange= -ones(1,Task.D_high);
        maxrange= ones(1,Task.D_high);
%         vars = minrange + ((vars+1)/2).*(maxrange-minrange) ;
 
        for i = 1:Task.D_high
            if vars(1,i)<minrange(1,i)
                vars(1,i) = minrange(1,i);
            end
            if vars(1,i)> maxrange(1,i)
                vars(1,i) = maxrange(1,i);
            end
        end
%         vars = (vars+1)/2*(100-0.001)+0.001;
    else % non-random embedding
        vars = rnvec(1:Task.D_high);
%         minrange = -ones(1,Task.D_high);
%         maxrange = ones(1,Task.D_high);
%         y=maxrange-minrange;
%         vars = y.*vars -1; 
        
%         vars = (vars+1)/2*(100-0.001)+0.001;
    end
 
     
    if rand(1)<=p_il
%         [x,objective,exitflag,output] = fminunc(Task.fnc,vars,options);
        nvars= (x-minrange)./y;
        m_nvars=nvars;
        m_nvars(nvars<0)=0;
        m_nvars(nvars>1)=1;
        if ~isempty(m_nvars~=nvars)  
            nvars=m_nvars;
            x=y.*nvars + minrange;
            objective=Task.fnc(x);
        end
        rnvec(1:d)=nvars;
        funcCount=output.funcCount;
    else
        x=vars;
        objective=Task.fnc(x);
        funcCount=1;
    end
end