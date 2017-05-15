package redeneural;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author Tiago
 */
public class LeitorCSV {

    public ArrayList<Double[][]> obterDados(int nEntrada, int nSaida) {

        String arquivoCSV = "arquivo.csv";
        BufferedReader br = null;
        String linha = "";
        String csvDivisor = ";";
        ArrayList dados = new ArrayList();
        try {

            br = new BufferedReader(new FileReader(arquivoCSV));
            br.readLine();
            while ((linha = br.readLine()) != null) {

                String[] dadosLinha = linha.split(csvDivisor);
                dados.add(dadosLinha);

            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        Util u = new Util();
        return u.converterArrayParaDouble(dados , nEntrada, nSaida);
    }

}
