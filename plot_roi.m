 
 t_roi = 35000; % concerning time (ms)
 
 % plot
 subplot(3,1,1);
 plot(firings_1(:,1),firings_1(:,2),'.');
 xlabel('time(s)');
 ylabel('Number of firing neurons');
 %plot(firings_1(:,1)/100/1000,firings_1(:,2),'.');
 str1 = strcat(num2str(Ne_1),{' Neurons with excitability ('},num2str(eta_zero_N),{', '},num2str(del_N),{') and noise '},num2str(sig));
 title(str1)
 c1 = xline(t_roi,'--r','linewidth',1.5);
 
 subplot(3,1,2);
 plot(+II_G(1,:));
 axis([0 num 0 1]);
 xlabel('time(s)');
 ylabel('Synaptic conductance');
 c2 = xline(t_roi,'--r','linewidth',1.5);
 
% plot v_1 distribution of interest
subplot(3,1,3);
% v_all_part = v_all(40001:40005,:);
histogram(v_all(t_roi,:),50,'facecolor','r');
xlabel('Membrane potential (mV)');
ylabel('Number of neurons');
str3 = strcat({'v_1 distribution at time '},num2str(t_roi/10000),{' s'});
title(str3)
