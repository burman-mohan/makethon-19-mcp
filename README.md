Run:
python .\server.py   

Add requirement file:
pip3 freeze > requirements.txt


const JSZip = require("jszip");
const fs = require("fs");
const path = require("path");

const zipFolder = (folderPath, zipFilePath) => {
  const zip = new JSZip();

  const addFilesToZip = (zipFile, folderPath, currentPath = "") => {
    const files = fs.readdirSync(path.join(folderPath, currentPath));

    for (const file of files) {
      const filePath = path.join(currentPath, file);
      const fullFilePath = path.join(folderPath, filePath);
      const stats = fs.statSync(fullFilePath);

      if (stats.isDirectory()) {
        addFilesToZip(zipFile, folderPath, filePath);
      } else {
        fileContent = fs.readFileSync(fullFilePath);
        zipFile.file(filePath, fileContent);
      }
    }
  };

  addFilesToZip(zip, folderPath);
  zip.generateAsync({ type: "nodebuffer" }).then((content) => {
    fs.writeFileSync(zipFilePath, content);
  }).catch((error) => console.log(error));;

  console.log(`Zip file created at: ${zipFilePath}`);
};

// Usage:
const folder = "D:\\Codes\\Spring\\test_workspace\\sla";
const zipName = "./myFolder-compressed.zip";

zipFolder(folder, zipName);
