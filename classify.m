function boolean = classify(A, beta)
    % A is de matrix die gebouwd is met build_A.m
    % beta is de vector die GD.m teruggeeft na het 'trainen'
    
    p = 1./(1+exp(-A*beta));
    boolean = ones(size(p));

    for i=1:length(p)
        if p(i) >= 0.5
            boolean(i) = 1;
        else
            boolean(i) = -1;
        end
    end
end