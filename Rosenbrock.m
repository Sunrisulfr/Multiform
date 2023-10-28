function obj = Rosenbrock(var,M,opt,D_eff)
%ROSENBROCK function
%   - var: design variable vector

    var = 5*var;
    opt = 5*opt;
    var = (M*(var-opt)')';
    var = var(1:D_eff);
    
    dim = length(var);
    
    sum = 0;
    for ii = 1:(dim-1)
        xi = var(ii);
        xnext = var(ii+1);
        new = 100*(xnext-xi^2)^2 + (xi-1)^2;
        sum = sum + new;
    end
    obj = sum  ;
end