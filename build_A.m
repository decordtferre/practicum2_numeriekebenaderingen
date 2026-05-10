function A = build_A(a1, a2, n)
    % a1: x1-coordinaat uit de DatasetCV
    % a1: x2-coordinaat uit de DatasetCV
    % n: de graad van de benadering, dus bijvoorbeeld n=1 is een rechte, 
    % n=2 ellips of parabool...

    A = ones(length(a1), 1); % eerste kolom is altijd volledig gevuld met enen

    for k=1:n
        A(:,2*k) = a1.^k;
        A(:,2*k+1) = a2.^k;
    end
end