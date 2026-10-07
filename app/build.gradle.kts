plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.savro.yoga"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.savro.yoga"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.activity:activity:1.8.2")
    implementation("androidx.webkit:webkit:1.11.0")
}

val syncWebAsset = tasks.register<Copy>("syncWebAsset") {
    from(rootProject.file("index.html"))
    into(layout.projectDirectory.dir("src/main/assets"))
    rename("index.html", "yoga.html")
}

// The faculty photos on the credits screen (images/*.jpg) - only edited via the root images/ folder.
val syncImages = tasks.register<Copy>("syncImages") {
    from(rootProject.file("images"))
    into(layout.projectDirectory.dir("src/main/assets/images"))
}

// Three.js (ES modules, loaded via the importmap in index.html) for the 3D pose avatar.
val syncThreeLibs = tasks.register<Copy>("syncThreeLibs") {
    from(rootProject.file("libs"))
    into(layout.projectDirectory.dir("src/main/assets/libs"))
}

// The rigged SMPL-X body mesh (GLB) the 3D avatar loads.
val syncAssets3d = tasks.register<Copy>("syncAssets3d") {
    from(rootProject.file("assets3d"))
    into(layout.projectDirectory.dir("src/main/assets/assets3d"))
}

// Real MoYo motion-capture axis-angle rotations for the 30 mocap-validated poses (fetched at
// runtime by index.html, relative to yoga.html - lives alongside it, not under assets3d/libs).
val syncPoseData = tasks.register<Copy>("syncPoseData") {
    from(rootProject.file("axis_angle_poses.json"))
    into(layout.projectDirectory.dir("src/main/assets"))
}

tasks.named("preBuild") {
    dependsOn(syncWebAsset, syncImages, syncThreeLibs, syncAssets3d, syncPoseData)
}
