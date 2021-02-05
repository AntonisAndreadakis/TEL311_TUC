function Vocabulary=GetVocabulary(vocfilename)
    %Get the vocabulary used in the term-documents from one of the files
	fid = fopen(vocfilename);
    VocabWords = strsplit(fgetl(fid), ';');
    fclose(fid);

    TotWords = length(strfind(VocabWords,',')); %Total Words in the Vocabulary
    WordBuffer=30;
    %Distinguish the different words from the string
    for i=1:TotWords %For every word
        tmpword=char(VocabWords(i));  %Convert word from cell to string
        tmpword(regexp(tmpword,'["]'))=[]; %Substitute the symbol "
        %sprintf('%d : %d - %s \n',TotWords, i,tmpword)
        for j=1:WordBuffer   %Allocate a maximum of 20 chars for each word
            word(i,j)=' '; %Initialize the word matrix
            %word(i,j) = tmpword;
        end
        if isempty(tmpword)
            word(i,:)=['' zeros(1,WordBuffer-size(tmpword,2))];
        else
            word(i,:) = [tmpword zeros(1,WordBuffer-size(tmpword,2))];
        end
    end
    Vocabulary=word(2:end,:);
end