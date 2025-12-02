// Imports table with credits to all third-party sources 
import LibrariesTable from "./LibrariesTable";
function About()
{
  
  return (
    <main >
      // Reviews Mission 
      <h2 >Mission</h2>
      <p >FairMap is a project intended to empower fairer decision making for congressional politicians, lobbyists, and staff to
      address real concerns in redistricting- most importantly the likelihood of gerrymandering. It is essential to understand 
      this tool is non-partisan and ethically envisioned, with all data its powered with sourced from open source libraries
      and academic research. FairMap does not take legal obligation for any maps checked for gerrymandering with FairMap's metric
      tools and is not a legal advisor. Opinions sourced from FairMap's data is at the discretion of the user's own opinion and 
      not FairMap.
      </p>

     // Reviews Scope of Application
      <h2>App</h2>
      <p > FairMap is specifically meant for scope of use in the Continental United States. The user may provide any valid
        map via shapefile or zip. FairMap uses a rich set of analytics including measures for compactness, partisanship, 
        competitiveness, and demographic bias. 
      </p>

     // Explains Processing of Data
      <h2>About the Data</h2>
      <p> Redistricting is process of dividing up a jurisdiction into districts for the purpose of electing representatives.
      Where and how the lines are drawn influences everything from who is likely to be elected to how resources get allocated.
      Gerrymandering makes it harder to have a voice in government.

      The official line-drawing process is different in every state, so as a result many of those are implictly assumed to be
      true for the sake of simplicity.

      It is highly recommended that you use MGGG for redistricting data if you do not have your own numbers to provide. 
      You may use hypotheticals to better determine information for demographic or election data.
      </p>
      
      // About the Team
      <h2>Team</h2> 

      <p ><b>Yasamean Zaidi-Dozandeh</b> is the project director and has over 8 years of experience in software development. She has 
          previously worked for Apple in technical troubleshooting and has multiple projects in the Javascript and Python sphere.
          She is working towards a Bachelors of Science in Computer Science, to graduate in Spring 2025.
         
      </p>
      <p ><b>Contributors & Honorable Mentions:</b> Sidney Mcclendon, Dr. Amlan Chatterjee</p>

     // Uses LibrariesTable to give Credit
      <h2 >Credits</h2>
      <p > FairMap is built off of a multitude of open source components including tools and metrics such as the following:</p>
      <LibrariesTable />


    </main>
  );
}
export default About;