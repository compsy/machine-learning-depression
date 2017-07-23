from learner.data_output.std_logger import L


class PlainTextExporter:

    @staticmethod
    def export(filename, text):
        L.info('Exporting plain text to: ' + filename)
        f = open('exports/' + filename, 'w')
        f.write(text)
        f.close()
