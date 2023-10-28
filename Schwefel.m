function obj = Schwefel(var,M,opt,D_eff)
%SCHWEFEL function
%   - var: design variable vector
    
    var = 500*var;
%     opt = 500*opt;
    
    opt = ones(1,length(var))*420.9687;
    
    var = ((var+opt)')';  
    var = var(1:D_eff);
    
    dim = length(var);

    sum = 0;
    for i = 1: dim
        sum = sum + var(i)*sin(sqrt(abs(var(i))));
    end
    
    obj = 418.9829*dim-sum;
    
    
    if obj<-10000
        disp('ddd');
    end
    
end

