function [b_mean_coeff,b_Z_coeff,b_P_coeff] = pls_modelweights(X,Y,ncomp)

    nboots = 5000;
    bs_b = bootstrp(nboots,@bootplsNCdim,X,Y,ncomp,'options',statset('UseParallel',true));
    r_bs_b = reshape(bs_b,nboots,1+size(X,2),[]);
    r_bs_b = r_bs_b(:,2:end,:);

    b_mean_coeff = squeeze(mean(r_bs_b,'omitnan'));
    b_ste_coeff = squeeze(std(r_bs_b,'omitnan'));
    b_ste_coeff(b_ste_coeff==0) = Inf;
    b_Z_coeff = b_mean_coeff ./ b_ste_coeff;
    b_P_coeff = 2*normcdf(-1*abs(b_Z_coeff),0,1);


    function BETA = bootplsNCdim(x,y,nc)
        [~,~,~,~,BETA] = plsregress(x,y,nc);
    end

end