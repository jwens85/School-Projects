SAS project for principal component analysis regarding factors present in insurance premium risk pricing. Risk factors are identified for use in a premium pricing model

FILENAME REFFILE '/folders/myfolders/CSV Dataset Claims.csv';
PROC IMPORT DATAFILE=REFFILE
	DBMS=CSV
	OUT=WORK.IMPORT;
	GETNAMES=YES;
RUN;
PROC CONTENTS DATA=WORK.IMPORT; RUN;
%web_open_table(WORK.IMPORT);

#Import data from work-import

proc princomp data=WORK.IMPORT plots(only ncomp=3)=(scree score (ellipse 
		alpha=0.05) pattern profile matrix);
	var auto_year age insured_zip months_as_customer policy_deductable 
		policy_annual_premium umbrella_limit total_claim_amount;
run;

#Principal component analysis of some possible factors of an insured member's risk profile
