package org.nd4j.linalg.versioncheck;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Created by Alex on 03/08/2017.
 */
@Slf4j
public class VersionCheck {

    public static final String SUPPRESS_VERSION_WARNING = "org.nd4j.versioncheck.suppress";

    private VersionCheck() {

    }

    public static void checkVersions(){

        boolean ignore = Boolean.parseBoolean(System.getProperty(SUPPRESS_VERSION_WARNING));
        if(ignore){
            return;
        }


        List<VersionInfo> l = getVersionInfo();

        Set<String> versions = new HashSet<>();

        for(VersionInfo vi : l){
            String version = vi.getVersion();
            if(version != null){
                versions.add(version);
            }
        }

        if(versions.size() > 1){
            log.warn("ND4J FAILED VERSION CHECK - {} DIFFERENT VERSIONS FOUND", versions.size());
            logVersions();
        }
    }

    public static void logVersions(){
        ServiceLoader<VersionInfo> sl = ServiceLoader.load(VersionInfo.class);
        Iterator<VersionInfo> iter = sl.iterator();

        List<VersionInfo> l = getVersionInfo();

        log.info("Found {} artifacts registered for ND4J version check", l.size());

        for(VersionInfo vi : l ){
            log.info(vi.getGAV());
        }
    }

    public static List<VersionInfo> getVersionInfo(){
        ServiceLoader<VersionInfo> sl = ServiceLoader.load(VersionInfo.class);
        Iterator<VersionInfo> iter = sl.iterator();
        List<VersionInfo> out = new ArrayList<>();
        while(iter.hasNext()){
            out.add(iter.next());
        }

        return out;
    }
}
