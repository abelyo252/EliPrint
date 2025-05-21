from eliprint import EliPrint

# Initialize with SQLite (default if MariaDB is not available)
recognizer = EliPrint(db_name="Eliprint")

# Add a song with extended metadata
recognizer.add_song(
    "songs/Tsehaytu Beraki - Bezay በዛይ.mp3",
    metadata={
        "title": "ዕምበባ | Embeba",
        "artist": "ሰለሞን ሃይለ | Solomon Haile",
        "album": " Embeba - Solomon Haile - Tigrigna Music",
        "lyrics": """አባ ሻውል ዝባን ጨርሒ ተወሊዓ ወርሒ
                    አባ ሻውል ጋር ከጨርሒ ጀርባ ጨረቃ በርታለች

                    ካብ ሰዓት ሸሞንተ ዕስራ ጎደል ተጸበኒ በል
                    ስምንት ሰዓት ከሀያ ጉዳይ ሲል ጀምረክ ጠብቀኝ

                    ወይዘሮ ሃና ኣደ ጽጌና ወዮ ትህባና ሃባና ዶንጊኸናና
                    ወይዘሮ ሃና÷ የጽጌ እናት ሊሰጡን ያሰቡትን ይስጡን አያቆዩን

                    ባዛይ ከመዓልካ ÷ ከመዓልካ ትብሉኒ እንታይ ክትደግሙኒ?
                    ባዛይ እንዴት ዋልክ ÷ እንዴት ዋልክ የምትሉኝ ድጋሜ ምን ልታደርጉኝ

                    ሰበይቲ ኣቦይ ዓቢዳ÷ ወርቂ ኽዳና ኣንዲዳ
                    የእንጀራ እናቴ አበደች÷ የወርቅ ልብሷም አቃጠለች

                    መን ኣንደዶ እንተበለት ባዘይ ባዘይ ፍልይ ኢልካ መሓዛይ
                    ማን አቃጠለው ስትል ባዘይ ባዘይ ለየት አልክ ጓዴ

                    አባ ሻውል ዝባን ጨርሒ ተወሊዓ ወርሒ
                    አባ ሻውል ጋር ከጨርሒ ጀርባ ጨረቃ በርታለች

                    ካብ ሰዓት ሸሞንተ ዕስራ ጎደል ተጸበኒ በል
                    ስምንት ሰዓት ከሀያ ጉዳይ ሲል ጀምረክ ጠብቀኝ

                    ወይኖየ ወይኖየ ወይኖ
                    የቀይ ዳማይ የቀይ ዳማየ

                    ድሕሪ ገዛይ የኳድድ ኣሎ
                    ከቤቴ ጀርባ እየጨፈረ ሄድ መለስ እያለ ነው

                    ሚኪኤለይ በዓል ዱር በረኻ
                    ሚካኤል ባለ ዱር በረሃ

                    ሓልወኒ ኣብ ዘለኻ ኣሊኻ
                    ጠብቀኝ የትም ብትሆን

                    ንሰብ እኳ ትብሎ ሓደራኻ
                    ለሰው እንኳን አደራ ይሰጣል አይደለም ላንተ

                    ሸላ ኣሎ ቆሊባ ኣሎ ክንድዚ ኩርሚዳ ኣሎ
                    ንስር አለ ÷ ነጣቂ አለ ÷ በፍጥነት የሚይዝ ሌባ አለ

                    ሓሪስካ ምብላዕ እኮ ኣሎ
                    አርሶ መብላትም አለ

                    አባ ሻውል ዝባን ጨርሒ ተወሊዓ ወርሒ
                    አባ ሻውል ጋር ከጨርሒ ጀርባ ጨረቃ በርታለች

                    ካብ ሰዓት ሸሞንተ ዕስራ ጎደል ተጸበኒ በል
                    ስምንት ሰዓት ከሀያ ጉዳይ ሲል ጀምረክ ጠብቀኝ

                    ወይዘሮ ሃና ኣደ ጽጌና ወዮ ትህባና ሃባና ዶንጊኸናና
                    ወይዘሮ ሃና÷ የጽጌ እናት ሊሰጡን ያሰቡትን ይስጡን አያቆዩን""",
        "history": """አባሸውል (ኤርትራ) ሠፈር ውስጥ አደይ ጸሃይቱ የማታ ስራ እየሰራች ባዘይን ትወደው ነበረ። 
                    ስትወደው በድብቅ ነው ባዛይ ሴተኛ አዳሪ እንደሆነች አያውቅም እናም እሱን ለማግኘት መሸት ያለ ሠዓት ትመርጣለች ሠው 
                    እንዳያያትና ለሡ ሥራዋን እንዳይነግሩት ትፈራ ነበር። አደይ ወይዘሮ ሀናን በግዜ ራትዋን ሰጥው እንዲሸኞት ተጠየቃለች ! 
                    በዛይ እንጀራ እናቱ አፍቅራው ብዙ ወርቅ አፍሳሰለታለች የሰፈሩ ሴት ሁሉ ባዛይ በዛይ ይለዋል።

                    አደይ ፀኻይቱ እና ባዘይ🥰""",
        "youtube_url": "https://www.youtube.com/watch?v=JGwWNGJdvx8",
        "picture_url": ""
    }
)

print("Your Song is recorded !")

recognizer.close()


from eliprint import EliPrint

# Initialize with SQLite (default if MariaDB is not available)
recognizer = EliPrint(db_name="Eliprint")


# Identify a song
result = recognizer.identify_song("songs/test/test1.mp3")

if result:
    print("Result : " , result)
else:
    print("Song not recognized")


