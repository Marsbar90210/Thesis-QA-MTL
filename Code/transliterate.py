from robobrowser import RoboBrowser as browser

def transliterate(text):
    br = browser(history=True, tries=1000)
    br.open('http://techwelkin.com/tools/transliteration/')
    
    form = br.get_form(id='converterForm')
    form['srcScript'].value = 'Roman'
    form['trgScript'].value = 'Devanagari'
    form['TextToConvert'].value = text
    br.submit_form(form)
    form = br.get_form(id='converterForm')
    return form['ConvertedText'].value

def de_transliterate(text):
    br = browser(history=True, tries=1000)
    br.open('http://techwelkin.com/tools/transliteration/')
    
    form = br.get_form(id='converterForm')
    form['srcScript'].value = 'Devanagari'
    form['trgScript'].value = 'Roman'
    form['TextToConvert'].value = text
    br.submit_form(form)
    form = br.get_form(id='converterForm')
    return form['ConvertedText'].value
